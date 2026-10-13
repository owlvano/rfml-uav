# Design

## Context

See proposal.md (Why) for motivation. Current state as observed in code:

- **Chunking** (`1_Data_Processing_Chunking.ipynb`):
  - `get_chunk_ids` runs separately on the L and H channels. The two windows are merged with `np.unique(np.concat((x_ids, y_ids)), axis=0)`, which sorts rows by start, and the same index pairs cut both channels. Only `arr_0` (the chunk list) is saved; starts are discarded.
  - `get_chunk_ids` is called without `min_packet_length`, so its default of 40 applies (the notebook-level `min_packet_length = chunk_size/2` is unused). Overlapping windows can therefore also arise within one channel.
  - Background (`00000`) uses contiguous non-overlapping windows.
- **WPD** (`2_Data_Processing_Wavelets.ipynb`): concatenates all `n` per BUI into `Wavelets/{bui}.csv` (L features ‖ H features, 256 columns). Recording identity is lost here.
- **Balancing** (`3_Data_Balancing.ipynb`): `RandomUnderSampler(sampling_strategy="majority")` over all chunks, written to `balanced/{feature}.npz` with `X, y` only.
- **Training** (`paper1-uav-identification/2_Classification_CNN.ipynb`): `StratifiedKFold(shuffle=True)` over balanced chunks. CE loss is computed on the softmax output. "Best fold" = highest test-fold accuracy, which then becomes `checkpoint_wpd_best.pt`, exported by `utils/export_onnx.py` (output `probs`).
- **CNN tail**: `fc4 → bn7 → prelu4 → softmax`. **RNN tail**: `fc4 → bn4 → softmax`.
- **Paper 2**: all paper2 notebooks read `chunked/*.npz` via `["arr_0"]` only; none import `rfml_uav.nn`.

## Goals / Non-Goals

**Goals:**
- Make chunk provenance available end to end with minimal edits to existing notebooks.
- Quantify the leakage effect (chunk vs recording split, dedup on/off) with everything else held constant.
- Produce the Paper 3 source checkpoint (`checkpoints/paper3/`) chosen without touching test data.

**Non-Goals:**
- Changing the CNN architecture beyond removing softmax (e.g. dropping `bn7`/`prelu4` on the logits). That would break checkpoint compatibility; revisit only if training on logits misbehaves.
- Re-running or editing the Paper 1 / Paper 2 notebooks.
- PSD/STFT features, RNN retraining, hyperparameter search.

## Decisions

### D1. Persist starts in the existing `chunked/*.npz` and keep `arr_0`
Save with `np.savez_compressed(path, chunks, starts=chunk_ids[:, 0])`. The positional array is still `arr_0`, so all Paper 2 readers keep working, and regenerated `arr_0` is identical because the window logic is unchanged.
- *Alternative:* a separate `chunked_meta/` directory. Rejected: more files, and they could drift out of sync.

### D2. Flag duplicates; do not drop them at chunking
Dedup happens at feature time, as an `is_dup` column. It is not applied in `chunked/`.
- Dropping windows in `chunked/` would silently change Paper 2's inputs while that paper is under review.
- A flag also lets the baseline report "with vs without dedup" from one feature file.

Rule: walk windows in start order and keep a window unless it overlaps the last *kept* window by more than `chunk_size/2`. This is one function in `src/rfml_uav/data/chunking.py` (`flag_duplicates(starts, chunk_size, max_overlap=0.5) -> bool mask`), so DroneDetect chunking in Phase 1 can reuse it.
- *Alternative:* IoU against all kept windows. Rejected: sorted input makes last-kept sufficient.

### D3. One feature file with metadata, legacy CSVs kept
The WPD notebook writes `Wavelets/features.npz` with `X (N,256) float32`, `bui (N,) str`, `n`, `chunk_idx`, `start`, `is_dup`, computed in the same loop that already writes `{bui}.csv`. Legacy CSVs stay so `3_Data_Balancing` and `onnx_inference.py` keep working.

### D4. Balancing inside the baseline notebook, carrying metadata
Background is undersampled in the new notebook using `RandomUnderSampler(...).sample_indices_` to index `X` together with the metadata arrays. The undersampling happens before splitting, mirroring Paper 1, so that the chunk-vs-recording comparison differs only in the splitter.
- *Alternative:* class-weighted CE on unbalanced data. Rejected for this change because it adds a second variable to the leakage comparison.

The undersampling seed equals the run seed.

### D5. Split API in `src/rfml_uav/eval/splits.py`
```python
def make_splits(y, groups, mode: Literal["chunk", "recording"], n_folds=5, val_frac=0.2, seed=0)
    -> list[tuple[train_idx, val_idx, test_idx]]
```
- **Outer test folds:** `StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)` in recording mode; `StratifiedKFold` in chunk mode.
- **Val:** carved from each outer train with one inner fold of the same splitter type (`n_splits=round(1/val_frac)`), so val is recording-disjoint in recording mode.
- **Groups:** `groups = [f"{b}_{n}"]`.

Plain function, no class. It also lives in src because Phase 1 (DroneDetect) and later phases reuse it.

### D6. Model selection
- Train a fixed 50 epochs (as Paper 1) with LR 0.01 and Adam.
- Keep the epoch state with the best val macro-F1 per fold, and evaluate it on test.
- For export, choose the fold with the best val macro-F1 under seed 0, recording mode, dedup on.
- Early stopping is skipped; best-val-epoch tracking gives the same protection with less code.

### D7. Logits + export wrapper
- Delete the `F.softmax` lines in `cnn.py` / `rnn.py`; `torch.max(outputs, 1)` in callers is unaffected.
- `export_onnx.py` exports `nn.Sequential(model, nn.Softmax(dim=1))` with unchanged names `input`/`probs`.
- Paths: parameterise ckpt/onnx path via argparse defaults pointing at the current v2 paths, so the same script exports `checkpoints/paper3/` models.

### D8. Metrics in `src/rfml_uav/eval/metrics.py`
One function `classification_report_dict(y_true, y_pred, probs, class_names)` returns:
- accuracy, macro-F1;
- per-class P/R/F1;
- the confusion matrix;
- max-prob arrays for correct and wrong predictions.

Aggregation (mean ± std over seed × fold) stays in the notebook with pandas `groupby`. Phases 4–7 reuse the function.

### D9. Run grid and storage
- Grid: 2 modes × 2 dedup settings × 5 seeds × 5 folds = 100 trainings on 256-d features.
- Per-run predictions (`y_true, y_pred, probs, test groups`) go to `results/paper3/phase0/{mode}_{dedup}_s{seed}_f{fold}.npz`, so tables and figures re-render without retraining.
- The notebook skips runs whose file already exists (resume).

## Risks / Trade-offs

- [Recording-split accuracy drops sharply] → Expected and reportable. Per the plan's gate, the new model is the chapter-3 source model, and the drop is stated openly. JMM uses the same table.
- [Phantom has only 21 recordings, one BUI] → Each test fold still gets ~4 Phantom recordings, which is enough for stratification. The splitter asserts that every class appears in every test fold.
- [Undersampling before splitting changes per-recording chunk counts] → The same balanced set feeds both modes, so the comparison stays fair. The design notes this as a limitation.
- [Paper 1 notebook semantics change (CE on logits)] → Tag the pre-change commit `momlet-2025`, and note it in the paper1 README.
- [PReLU/BN on logits may limit logit range for the energy score] → Accepted for Phase 0. Check the energy-score spread in this notebook's confidence plots before Phase 6.
- [Regenerating `chunked/` overwrites existing files] → `arr_0` stays identical by construction. Verify by checksum on a few recordings before and after.

## Migration Plan

1. Tag the current HEAD `momlet-2025` (local tag; pushing is up to the author).
2. Back up `DATA_PATH/chunked` and `DATA_PATH/Wavelets` (or rely on regeneration from `raw/`).
3. Re-run the chunking and WPD notebooks, then verify `arr_0` checksums.
4. Run the baseline notebook.

Rollback: revert the commit; the legacy data files are unchanged in format.
