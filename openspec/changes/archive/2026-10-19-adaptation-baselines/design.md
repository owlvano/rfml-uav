# Design

## Context

See proposal.md (Why). The design builds on:

- **Models.** 25 Phase 1 models, `checkpoints/paper3/phase1/active_linear_s{seed}_f{fold}.pt`, loaded with `get_model("CNN", 256, 7, linear_tail=True)`.
  - The CNN has 6 `BatchNorm1d` layers (bn1–bn6): 410k parameters, of which 2,016 are BN affine and 903 are the `fc4` head.
- **Rows.**
  - Phase 1 `runs/active_linear_s*_f*.npz` store `test_idx`: a class-balanced subset of the model's CLEAN test recordings, about 19 recordings and 14k rows.
  - Active rows are capped at 1,000 per recording, so whole recordings give near-balanced classes.
- **Recording counts.**
  - Shift A: BLUE has 95 active recordings, WIFI and BOTH 100 each. DIS has 10 per subset; PHA has 10 in BLUE and 15 (including FY) in WIFI and BOTH.
  - Shift B: a model's CLEAN test fold has 2–3 recordings per class (2 for DIS and PHA).
- **Impaired features.** `results/paper3/phase2/dronedetect/{kind}_{s}.npz` (`X`, `row`) covers all 95k CLEAN active rows and is row-aligned with `features.npz`.
- **Shared code.** `train_best_val` and `predict_logits` (`nn/train.py`), `paired_test` (`eval/stats.py`).

## Goals / Non-Goals

**Goals:**
- Baseline implementations small enough to read in one sitting, with no dependency beyond torch. OG-TCST's stream notebook must be able to call them on any feature batch.
- A protocol whose only use of labels is scoring and dev-shift tuning.

**Non-Goals:**
- SOTA test-time adaptation variants (EATA, SAR, CoTTA, SHOT). Three classic baselines cover the families: statistics-only, entropy on BN, self-training. A reviewer asking for more can be pointed to TENT and SHOT-like PL.
- Class-balanced or curriculum pseudo-labelling. "Naive" is deliberate: it is the lower bound that OG-TCST's gate is meant to fix.

## Decisions

### D1. Module `rfml_uav/adapt/baselines.py`: three functions, `(model, X, **hp) -> (adapted, info)`
- **Input:** `X` is a float tensor of shape `(N, 1, 256)` already on the model's device. No labels: the signature makes that impossible.
- **Output:** the function deep-copies the model, adapts the copy, and returns it in `eval()` mode, together with `info = {"fwd_per_window", "bwd_per_window", "n_updated", ...}`.
- **Alternative:** a class hierarchy with a `fit/predict` interface. Rejected: 3 stateless functions with no shared state don't need one.

### D2. AdaBN: one pass in train mode with cumulative statistics
- Set every BN layer's `momentum=None` (PyTorch then keeps a cumulative average) and call `reset_running_stats()`.
- Run `model.train()` under `torch.no_grad()` over `np.array_split` batches of ~1024 windows. Near-equal batch sizes make the cumulative mean match the adapt-set mean within 1e-4.
- Restore the momentum afterwards.
- Cost: 1 forward pass per window.
- The running variance is the average within-batch variance. That is what BN computes in train mode and what the original AdaBN implementations do.
- **Alternative:** exact per-layer pooled statistics (a pass per layer). Rejected: 6× the forward cost for a variance difference of < 1% at batch 1024.

### D3. TENT (Wang et al., 2021), offline variant
- Only BN `weight` and `bias` require gradients. Training runs in train mode, so BN uses batch statistics, as in TENT.
- Loss: mean softmax entropy. Optimiser: Adam, batch 256, shuffled with a seeded generator. E epochs over the adapt set.
- Afterwards, run the D2 statistics pass so eval-mode scoring uses statistics from the adapted network.
- Cost: E + 1 forward and E backward passes per window; `n_updated` = 2,016.

### D4. Naive pseudo-labelling
- One eval-mode source pass over the adapt set gives the argmax and max-softmax of each window. Labels are fixed from that point on.
- Keep the windows with max-softmax ≥ τ. If fewer than 256 pass, return the source copy with `info["selected"] = 0`.
- Otherwise fine-tune all parameters with cross-entropy for E epochs: Adam, batch 256, train mode. BN running statistics update with the default momentum, which is part of "naive".
- Cost: `1 + E·k/N` forward and `E·k/N` backward passes per window, where k windows are selected; `n_updated` = all parameters.
- `info` also records `selected_per_class`.

### D5. Determinism
- Each function takes a `seed`. Inside it: `torch.manual_seed(seed)` plus a seeded `torch.Generator` for shuffling.
- The notebook sets `torch.backends.cudnn.deterministic = True` and `benchmark = False`, since Conv1d's backward pass is otherwise non-deterministic on GPU.
- The self-check runs on CPU.

### D6. Adapt/test split: per class, half the recordings, seeded by (seed, fold, target)
- Group the target recordings by drone. For each class, permute with `default_rng([seed, fold, TARGET_ID[target]])`; `floor(n/2)` go to adapt, the rest to test. Classes with fewer than 2 recordings raise an error. They can't occur here (minimum 2), but the assert documents the assumption.
- **Why not `StratifiedGroupKFold(2)`?** It doesn't guarantee that both halves get every class when a class has only 2 groups, and per-class permutation is simpler.
- Rows:
  - shift A: every active row of the target's recordings, minus unseen (drone, mode);
  - shift B: every CLEAN active row of the model's test recordings (not only the balanced `test_idx` subset), taken from the impaired file.
- Splits are saved to `phase3/splits/{target}_s{seed}_f{fold}.npz` (`adapt_rows`, `test_rows`) for E5-lite and OG-TCST to reuse.

### D7. Dev-shift hyperparameter grid (seed-0 models, 5 folds)

| method | grid | size |
|---|---|---|
| AdaBN | none | 1 |
| TENT | lr ∈ {1e-4, 1e-3, 1e-2} × epochs ∈ {1, 3} | 6 |
| PL | τ ∈ {0.8, 0.9, 0.95} × lr ∈ {1e-4, 1e-3} × epochs ∈ {1, 3} | 12 |

- Dev cells: awgn 1, gain 2, adc 2, tilt 3, split by D6 like the targets.
- Score: mean macro-F1 gain over source-only on the dev test halves, averaged over 4 cells × 5 folds.
- Tie-break (< 0.1 pp): fewer backward passes per window, then lower lr.
- The choice is written to `phase3/selected_hparams.json` before the target loop. The target loop reads only that file.
- In total: 4 × 5 × 19 = 380 adaptations on about 9k-window adapt sets. Minutes.

### D8. Run storage and report
- One `phase3/runs/{target}_s{seed}_f{fold}.npz` per run holds:
  - `test_rows` and `clean_logits` (shift B only);
  - `{method}_logits` for source, adabn, tent and pl;
  - `{method}_info` as a JSON string, with wall time measured with `torch.cuda.synchronize()`.
- Tables are rebuilt from the run files alone:
  - `e3_table.csv`: macro-F1, accuracy, gain [CI, p], recovered fraction, best-baseline flag;
  - `e3_auroc.csv`;
  - `e3_cost.csv`;
  - `e3_recovery.png`: recovered fraction per target and method, grouped as shift A | shift B.
- Recovered fraction = (adapted − source) / (clean − source), averaged per run and then over runs.
  - Runs where clean − source is below 1 pp are excluded from that average and counted. That can happen for a single fold on a mild target.

## Risks / Trade-offs

- **[Shift-B adapt sets are small]:** about 9k windows from 1–2 recordings per class, which can bias the AdaBN statistics. → Inherent to a leak-free protocol: only test recordings are unseen. State it, and add one sanity row: AdaBN on the clean adapt half should give a gain of ≈ 0. A clearly negative gain flags statistics noise.
- **[PL / TENT collapse]:** with a high lr, everything gets predicted as one class. → The dev grid includes conservative settings and would pick them. `selected_per_class` and the per-class F1 expose collapse.
- **[Dev cells are mild]:** 0.5–6 pp drops give a weak tuning signal, so the tuning may favour "do little". → Acceptable and conservative: under-tuning hurts the baselines, not OG-TCST. A 2-line note in the paper covers it, and the same grid and dev cells will select OG-TCST's own hyperparameters, so the comparison stays fair.
- **[Shift A test halves come from one session]:** the five recordings per condition are likely consecutive. → Already a stated limitation (Phase 1). The adapt/test halves are still recording-disjoint.
- **[GPU wall time is not node energy]** → Labelled as a proxy. E9 measures joules.

## Migration Plan

Additive only: a new package directory, a new notebook and `phase3/` results. Rollback = delete them.

## Open Questions

- RQ numbering: the proposal assumes E3 serves RQ2. Confirm against the planning doc; this doesn't change the work.
