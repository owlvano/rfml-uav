"""Label-free stream adaptation (thesis 3.3): one chunked prequential runner for every method.

Per chunk of active windows: predict with the current state -> energy gate -> temporal vote -> buffer the
accepted windows -> drift trigger -> maybe adapt. Every prediction therefore precedes any update it feeds.
No function here takes labels. Costs are counted in window-passes (forward / backward).
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rfml_uav.adapt.baselines import _batches, _bn_layers, adabn

METHODS = ("source", "triggered", "continuous", "periodic", "tent", "pl", "vote_pl", "vote_pl_head",
           "balanced", "gated_continuous", "gated_balanced")
GATED = {"gated_continuous", "gated_balanced"}  # receiver gate on the noise-floor drift (Phase 6)
BALANCED = {"balanced", "gated_balanced"}  # statistics from the drone-balanced buffer
TRIGGERED = {"triggered", "vote_pl", "vote_pl_head"}
REFRESHING = TRIGGERED | {"periodic"}  # methods with refresh events: gate and trigger re-anchor after each
HP = dict(M=250, W=25, q=0.1, theta=3.0, R=4, C=8, B=1000, D=3, lam=0.5,  # D1-D5
          m=0.1, P=20, lr=1e-4, tau=0.9, rho=0.8, batch_size=256, gate=True,  # D6; lr is per method
          phi_on=0.5, h=0.5, Bc=200)  # Phase 6: receiver gate (on > phi_on, off < h * phi_on), per-class buffer


def energy(z):
    return -torch.logsumexp(z, 1)


def conv_bns(model):
    """BN layers registered before the first Linear (bn1-bn3 of the CNN): the restricted-refresh scope."""
    out = []
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            break
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            out.append(mod)
    return out


def n_stats(bns):
    return sum(2 * bn.num_features for bn in bns)


def vote(prev, preds, W, n_classes):
    """Sliding majority of the last W predictions; prev = the up-to-(W-1) earlier ones. Ties -> most recent tied class."""
    s = np.r_[np.asarray(prev, np.int64), np.asarray(preds, np.int64)]
    oh = np.eye(n_classes, dtype=np.int64)[s]
    cs = np.r_[np.zeros((1, n_classes), np.int64), oh.cumsum(0)]
    last = np.maximum.accumulate(np.where(oh, np.arange(len(s))[:, None], -1))
    p = np.arange(len(prev), len(s))
    counts = cs[p + 1] - cs[np.maximum(p + 1 - W, 0)]
    return (counts * (len(s) + 1) + last[p]).argmax(1)


def block_labels(preds, W, rho, n_classes):
    """Vote-PL labels: majority of each consecutive W-block of predictions; blocks with agreement < rho -> -1."""
    out = np.full(len(preds), -1)
    for i in range(0, len(preds), W):
        c = np.bincount(preds[i:i + W], minlength=n_classes)
        if c.max() >= rho * len(preds[i:i + W]):
            out[i:i + W] = c.argmax()
    return out


def refresh(model, source_conv, Xb, votes, D, lam, seed=0):
    """Guarded BN-statistics refresh of `model` in place -> (kind, statistics changed).

    Full (every BN takes the buffer statistics) if the buffer's votes cover >= D classes; otherwise restricted:
    conv BNs set to (1 - lam) * source + lam * buffer, the FC BNs keep their current statistics.
    """
    new = adabn(model, Xb, seed=seed)[0]
    if len(np.unique(votes)) >= D:
        for bn, nb in zip(_bn_layers(model), _bn_layers(new)):
            bn.running_mean.copy_(nb.running_mean)
            bn.running_var.copy_(nb.running_var)
        return "full", n_stats(_bn_layers(model))
    for bn, nb, (mu0, var0) in zip(conv_bns(model), conv_bns(new), source_conv):
        bn.running_mean.copy_((1 - lam) * mu0 + lam * nb.running_mean)
        bn.running_var.copy_((1 - lam) * var0 + lam * nb.running_var)
    return "restricted", n_stats(conv_bns(model))


def _chunk_ref(e, M):
    """(mean, std) of the means of consecutive M-chunks of energies."""
    means = [c.mean() for c in np.array_split(e, max(len(e) // M, 1))]
    return float(np.mean(means)), float(np.std(means))


def run_stream(model, X, active, method, hp, calib, seed=0, drift=None):
    """Run one time-ordered stream. X: centred model inputs for all windows; active: bool flags (only active
    windows are classified); hp: overrides of HP; calib: source energies on its own CLEAN validation rows;
    drift: optional per-window noise-floor drift (stream_normalise(..., return_drift=True)), read by gated methods.

    Returns per-active-window pred, vote, energy, known; `cost` (pass counts, events, refreshes, statistics or
    parameters changed per event, event positions in active windows) and `buffers` (buffer indices per event).
    """
    assert method in METHODS, method
    hp = {**HP, **hp}
    M, W, q = hp["M"], hp["W"], hp["q"]
    torch.manual_seed(seed)
    model = copy.deepcopy(model).eval()
    dev = next(model.parameters()).device
    Xa = torch.as_tensor(X, dtype=torch.float32, device=dev)[torch.as_tensor(np.asarray(active, bool), device=dev)]
    n = len(Xa)
    calib = np.asarray(calib, np.float64)
    assert len(calib) >= 2 * M, "calibration needs at least two chunks"
    source_conv = [(bn.running_mean.clone(), bn.running_var.clone()) for bn in conv_bns(model)]

    params = None
    if method == "tent":
        params = [p for bn in _bn_layers(model) for p in (bn.weight, bn.bias) if p is not None]
    elif method == "pl":
        params = list(model.parameters())
    elif method == "vote_pl_head":
        params = list([m for m in model.modules() if isinstance(m, nn.Linear)][-1].parameters())
    elif method == "vote_pl":
        params = list(model.parameters())
    if params is not None:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
    if method in ("tent", "pl"):
        opt = torch.optim.Adam(params, lr=hp["lr"])
    if method in ("continuous", "balanced") or method in GATED:
        for bn in _bn_layers(model):
            bn.momentum = hp["m"]
    source_bn = [(bn.running_mean.clone(), bn.running_var.clone()) for bn in _bn_layers(model)]
    da = None if drift is None else np.nan_to_num(np.asarray(drift, np.float64)[np.asarray(active, bool)])
    on, cbuf, on_n = method == "balanced", {}, 0  # cbuf: vote class -> last Bc accepted window indices

    pred, vt = np.zeros(n, np.int64), np.zeros(n, np.int64)
    en, known = np.zeros(n, np.float32), np.zeros(n, bool)
    buf = np.zeros(0, np.int64)
    thr, ref = np.quantile(calib, 1 - q), _chunk_ref(calib, M)
    anchor, cool = None, 0  # anchor: (energies, chunk means) collected after a refresh, None when anchored
    cost = dict(fwd=n, bwd=0, events=0, full_refreshes=0, restricted_refreshes=0, changed_per_event=[], event_at=[],
                on=[])
    buffers, n_classes = [], None

    for k, lo in enumerate(range(0, n, M)):
        hi = min(lo + M, n)
        with torch.no_grad():
            z = model(Xa[lo:hi])
        n_classes = z.shape[1]
        e = energy(z).cpu().numpy()
        p = z.argmax(1).cpu().numpy()
        ok = e <= thr if hp["gate"] else np.ones(len(e), bool)
        pred[lo:hi], en[lo:hi], known[lo:hi] = p, e, ok
        vt[lo:hi] = vote(pred[max(lo - W + 1, 0):lo], p, W, n_classes)
        idx = lo + np.flatnonzero(ok)
        buf = np.r_[buf, idx][-hp["B"]:]
        mean_k = e[ok].mean() if ok.any() else None

        if anchor is not None:
            anchor[0].append(e)
            if mean_k is not None:
                anchor[1].append(mean_k)
            if len(anchor[0]) == hp["R"]:
                thr = np.quantile(np.concatenate(anchor[0]), 1 - q)
                if len(anchor[1]) >= 2:
                    ref = (float(np.mean(anchor[1])), float(np.std(anchor[1])))
                anchor = None

        fire = False
        if method in TRIGGERED:
            if cool > 0:
                cool -= 1
            elif anchor is None and mean_k is not None and abs(mean_k - ref[0]) > hp["theta"] * ref[1]:
                fire, cool = True, hp["C"]
        elif method == "periodic":
            fire = (k + 1) % hp["P"] == 0

        if fire and len(buf) >= 2:
            changed = 0
            if method in ("vote_pl", "vote_pl_head"):
                lab = block_labels(pred[buf], W, hp["rho"], n_classes)
                sel = buf[lab >= 0]
                y = torch.as_tensor(lab[lab >= 0], device=dev)
                o = torch.optim.Adam(params, lr=hp["lr"])
                model.train()
                for b in _batches(len(sel), hp["batch_size"], 1, seed + k, dev):
                    o.zero_grad()
                    F.cross_entropy(model(Xa[torch.as_tensor(sel, device=dev)[b]]), y[b]).backward()
                    o.step()
                    cost["fwd"] += len(b)
                    cost["bwd"] += len(b)
                    changed = sum(x.numel() for x in params)
                model.eval()
            kind, s = refresh(model, source_conv, Xa[torch.as_tensor(buf, device=dev)], vt[buf], hp["D"], hp["lam"],
                              seed + k)
            cost["fwd"] += len(buf)
            cost[f"{kind}_refreshes"] += 1
            cost["events"] += 1
            cost["changed_per_event"].append(changed + s)
            cost["event_at"].append(hi)
            buffers.append(buf.copy())
            anchor = ([], [])

        if method in GATED and da is not None:  # after this chunk's predictions: updates follow the new state
            if not on and da[hi - 1] > hp["phi_on"]:
                on, cbuf = True, {}  # ON statistics come from post-change windows only
            elif on and da[hi - 1] < hp["h"] * hp["phi_on"]:
                on = False
                for bn, (mu0, var0) in zip(_bn_layers(model), source_bn):
                    bn.running_mean.copy_(mu0)
                    bn.running_var.copy_(var0)
        if method in GATED:
            cost["on"].append(int(on))
            on_n += (hi - lo) * on
        if method in BALANCED:
            for c in np.unique(vt[idx]):
                cbuf[c] = np.r_[cbuf.get(c, np.zeros(0, np.int64)), idx[vt[idx] == c]][-hp["Bc"]:]

        g = Xa[torch.as_tensor(idx, device=dev)]
        if method in BALANCED and on:
            full = [c for c in cbuf if len(cbuf[c]) >= hp["Bc"] / 2]
            if len(full) >= hp["D"]:
                g = Xa[torch.as_tensor(np.concatenate([cbuf[c] for c in full]), device=dev)]
                model.train()
                with torch.no_grad():
                    model(g)
                model.eval()
                cost["fwd"] += len(g)
                cost["events"] += 1
                cost["changed_per_event"].append(n_stats(_bn_layers(model)))
        elif (method == "continuous" or method == "gated_continuous" and on) and len(g) >= 2:
            model.train()
            with torch.no_grad():
                model(g)
            model.eval()
            cost["fwd"] += len(g)
            cost["events"] += 1
            cost["changed_per_event"].append(n_stats(_bn_layers(model)))
        elif method in ("tent", "pl"):
            if method == "pl":
                conf = F.softmax(z[ok], 1).max(1).values.cpu().numpy()
                g, y = g[conf >= hp["tau"]], torch.as_tensor(p[ok][conf >= hp["tau"]], device=dev)
            if len(g) >= 2:
                model.train()  # train-mode BN: TENT also moves the running statistics, as in Wang et al.
                out = model(g)
                loss = (-(F.softmax(out, 1) * F.log_softmax(out, 1)).sum(1).mean() if method == "tent"
                        else F.cross_entropy(out, y))
                opt.zero_grad()
                loss.backward()
                opt.step()
                model.eval()
                cost["fwd"] += len(g)
                cost["bwd"] += len(g)
                cost["events"] += 1
                cost["changed_per_event"].append(sum(x.numel() for x in params))

    cost["fwd_per_window"], cost["bwd_per_window"] = cost.pop("fwd") / n, cost.pop("bwd") / n
    cost["updates"] = cost["events"]
    cost["on_fraction"] = on_n / n if method in GATED else float(method == "balanced")
    return {"pred": pred, "vote": vt, "energy": en, "known": known, "cost": cost, "buffers": buffers}


if __name__ == "__main__":
    import inspect

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    centres = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])

    def blobs(n_blocks, block=200):
        """Recording-like stream: blocks of one class each, classes in random order; input shape (n, 1, 2)."""
        y = torch.as_tensor(np.repeat(rng.integers(0, 3, n_blocks), block))
        return (centres[y] + 0.5 * torch.randn(len(y), 2)).unsqueeze(1), y

    src = nn.Sequential(nn.Conv1d(1, 8, 2), nn.BatchNorm1d(8), nn.ReLU(), nn.Flatten(),
                        nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 3))
    Xs, ys = blobs(30)
    opt = torch.optim.Adam(src.parameters(), lr=0.01)
    for _ in range(300):
        b = torch.randint(0, len(Xs), (128,))
        opt.zero_grad()
        F.cross_entropy(src(Xs[b]), ys[b]).backward()
        opt.step()
    src.eval()
    before = copy.deepcopy(src.state_dict())
    with torch.no_grad():
        calib = energy(src(blobs(30)[0])).numpy()

    X0, y0 = blobs(30)                       # stationary prefix (6,000 windows)
    X1, y1 = blobs(30)
    X, y = torch.cat([X0, 3 * X1 + 2]), torch.cat([y0, y1])  # receiver-like shift at the change point
    cp = len(X0)
    active = rng.random(len(X)) < 0.9         # noise windows are skipped by the runner
    ya = y[active].numpy()
    cpa = int(active[:cp].sum())
    hp = dict(theta=4.0)
    run = {m: run_stream(src, X, active, m, {**hp, "lr": 1e-3}, calib) for m in METHODS}
    assert all(torch.equal(before[k], v) for k, v in src.state_dict().items()), "runner changed the source model"

    # no label argument
    assert list(inspect.signature(run_stream).parameters) == ["model", "X", "active", "method", "hp", "calib", "seed",
                                                               "drift"]

    # source-only cost: one forward per window, nothing else
    c = run["source"]["cost"]
    assert c["fwd_per_window"] == 1.0 and c["bwd_per_window"] == 0.0 and c["events"] == 0, c
    assert len(run["source"]["pred"]) == active.sum()

    # vote: majority, short history, ties -> most recent
    assert vote([], [0, 1, 1], 3, 3).tolist() == [0, 1, 1]
    assert vote([2, 2], [0], 3, 3).tolist() == [2]
    assert vote([0], [1], 2, 3).tolist() == [1]
    assert vote([1, 1, 2, 2], [2], 3, 3).tolist() == [2]   # window = last 3 only
    p = run["source"]["pred"]
    assert np.array_equal(run["source"]["vote"], np.concatenate([vote(p[:i][-24:], p[i:i + 1], 25, 3) for i in range(len(p))]))

    # gate: rejection stays near q on a stationary stream, flat over time
    Xst, _ = blobs(100)
    r = ~run_stream(src, Xst, np.ones(len(Xst), bool), "source", {}, calib)["known"]
    assert abs(r.mean() - HP["q"]) < 0.03 and abs(r[:len(r) // 2].mean() - r[len(r) // 2:].mean()) < 0.03, r.mean()

    # stable stream: no events; shift: the trigger fires after the change point, not before
    assert run_stream(src, Xst, np.ones(len(Xst), bool), "triggered", hp, calib)["cost"]["events"] == 0
    t = run["triggered"]
    assert t["cost"]["events"] >= 1 and t["cost"]["event_at"][0] > cpa, (t["cost"]["event_at"], cpa)
    assert t["cost"]["event_at"][0] <= cpa + 3 * HP["M"]

    # prediction precedes update: up to the first event, windows were scored by the source state
    first = t["cost"]["event_at"][0]
    assert np.array_equal(t["energy"][:first], run["source"]["energy"][:first])
    assert not np.array_equal(t["energy"][first:], run["source"]["energy"][first:])

    # gated windows never enter a buffer
    for m in REFRESHING:
        assert all(run[m]["known"][b].all() for b in run[m]["buffers"]), m

    # cooldown: with a hair trigger, events are > C chunks apart
    c = run_stream(src, X, active, "triggered", dict(theta=0.01), calib)["cost"]
    gaps = np.diff(c["event_at"]) // HP["M"]
    assert c["events"] >= 3 and gaps.min() > HP["C"], gaps

    # guarded refresh: single-class buffer -> restricted (conv BN only, lambda blend); diverse -> full
    Xb = 3 * blobs(5)[0][:1000] + 2
    m1 = copy.deepcopy(src)
    conv0 = [(b.running_mean.clone(), b.running_var.clone()) for b in conv_bns(m1)]
    kind, s = refresh(m1, conv0, Xb, np.zeros(len(Xb), int), D=3, lam=0.5)
    ref_stats = adabn(src, Xb)[0]
    assert kind == "restricted" and s == 2 * 8
    assert torch.allclose(m1[1].running_mean, 0.5 * src[1].running_mean + 0.5 * ref_stats[1].running_mean)
    assert torch.equal(m1[5].running_mean, src[5].running_mean) and torch.equal(m1[5].running_var, src[5].running_var)
    m2 = copy.deepcopy(src)
    kind, s = refresh(m2, conv0, Xb, np.arange(len(Xb)) % 3, D=3, lam=0.5)
    assert kind == "full" and s == 2 * (8 + 16)
    assert all(torch.allclose(a.running_mean, b.running_mean) for a, b in zip(_bn_layers(m2), _bn_layers(ref_stats)))
    for mm in (m1, m2):
        assert all(torch.equal(p, before[k]) for k, p in mm.named_parameters())

    # vote-PL labels: block majorities kept only where agreement >= rho
    assert block_labels(np.array([0, 0, 1, 2, 2, 2, 1]), 3, 0.6, 3).tolist() == [0, 0, 0, 2, 2, 2, 1]
    assert block_labels(np.array([0, 1, 2]), 3, 0.6, 3).tolist() == [-1, -1, -1]

    # every method runs; backward passes only for the gradient methods
    for m, o in run.items():
        assert (o["cost"]["bwd_per_window"] > 0) == (m in ("tent", "pl", "vote_pl", "vote_pl_head")), (m, o["cost"])
    assert run["periodic"]["cost"]["events"] == len(range(0, len(ya), HP["M"])) // HP["P"]

    def f1(m):  # target-segment vote accuracy (labels used only here, for the check)
        return (run[m]["vote"][cpa:] == ya[cpa:]).mean()

    assert f1("triggered") > f1("source") + 0.1, (f1("source"), f1("triggered"))
    # determinism
    assert np.array_equal(run_stream(src, X, active, "vote_pl", {**hp, "lr": 1e-3}, calib)["vote"], run["vote_pl"]["vote"])
    # --- Phase 6: receiver gate and drone-balanced buffer ---
    n_all, srun = len(X), run["source"]
    for m in GATED:  # stable floor (or no drift at all): gated == source, no updates, one forward per window
        for d in (np.zeros(n_all), None):
            o = run_stream(src, X, active, m, {}, calib, drift=d)
            assert np.array_equal(o["vote"], srun["vote"]) and o["cost"]["updates"] == 0, m
            assert o["cost"]["fwd_per_window"] == 1.0 and o["cost"]["on_fraction"] == 0.0, m
    # hysteresis: ON above phi_on, stays ON between h * phi_on and phi_on, OFF below; OFF restores the source
    pos = np.flatnonzero(active)
    d = np.zeros(n_all)
    d[pos[cpa]:] = 1.0
    d[pos[cpa + 4 * HP["M"]]:] = 0.3        # between 0.25 and 0.5: stays ON
    off_at = cpa + 12 * HP["M"]
    d[pos[off_at]:] = 0.1                   # below 0.25: OFF
    o = run_stream(src, X, active, "gated_continuous", {}, calib, drift=d)
    on = np.array(o["cost"]["on"])
    k_on, k_off = cpa // HP["M"], off_at // HP["M"]
    assert not on[:k_on].any() and on[k_on + 1:k_off].all() and not on[k_off + 1:].any(), on
    assert o["cost"]["updates"] > 0 and o["cost"]["bwd_per_window"] == 0
    after = (k_off + 1) * HP["M"]           # the chunk after the OFF switch is predicted by the source state again
    assert np.allclose(o["energy"][after:], srun["energy"][after:], atol=1e-5)
    assert not np.allclose(o["energy"][cpa + HP["M"]:off_at], srun["energy"][cpa + HP["M"]:off_at], atol=1e-5)
    # balanced buffer: a single-drone stream never updates; per-class cap bounds every update batch
    Xone = (centres[torch.zeros(6000, dtype=torch.long)] + 0.5 * torch.randn(6000, 2)).unsqueeze(1)
    o = run_stream(src, Xone, np.ones(6000, bool), "balanced", {}, calib)
    assert o["cost"]["updates"] == 0 and o["cost"]["fwd_per_window"] == 1.0
    o = run["balanced"]["cost"]
    assert o["updates"] > 0 and o["bwd_per_window"] == 0
    per_update = (o["fwd_per_window"] - 1) * len(ya) / o["updates"]
    assert per_update <= 3 * HP["Bc"], per_update
    print({m: round(f1(m), 3) for m in METHODS})
    print({m: (round(o["cost"]["fwd_per_window"], 2), round(o["cost"]["bwd_per_window"], 2), o["cost"]["events"])
           for m, o in run.items()})
    print("stream adaptation self-check OK")
