"""Label-free adaptation baselines: (model, X, **hp) -> (adapted copy, info). X: unlabeled target features."""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bn_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def _forward(model, X, batch_size):
    with torch.no_grad():
        return torch.cat([model(X[i:i + batch_size]) for i in range(0, len(X), batch_size)])


def _stats_pass(model, X, batch_size, seed=0):
    """Re-estimate every BN running mean/var on X in one train-mode pass (cumulative average over near-equal batches).

    Batches are a seeded shuffle: X is often sorted by recording, and contiguous batches would drop the
    between-recording variance from the running variance.
    """
    bns = _bn_layers(model)
    momenta = [bn.momentum for bn in bns]
    for bn in bns:
        bn.momentum = None  # cumulative moving average
        bn.reset_running_stats()
    model.train()
    with torch.no_grad():
        perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed)).to(X.device)
        for chunk in torch.tensor_split(perm, -(-len(X) // batch_size)):
            model(X[chunk])
    for bn, m in zip(bns, momenta):
        bn.momentum = m
    return model.eval()


def _batches(n, batch_size, epochs, seed, device):
    """Shuffled index batches for `epochs` passes; drops the last partial batch (BatchNorm needs > 1 sample)."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n - batch_size + 1, batch_size):
            yield perm[i:i + batch_size]


def adabn(model, X, batch_size: int = 1024, seed: int = 0):
    """AdaBN: BN statistics re-estimated on target data; no trainable parameter changes."""
    torch.manual_seed(seed)
    m = _stats_pass(copy.deepcopy(model), X, batch_size, seed)
    return m, {"fwd_per_window": 1.0, "bwd_per_window": 0.0, "n_updated": 0}


def tent(model, X, lr: float, epochs: int, batch_size: int = 256, seed: int = 0):
    """TENT (Wang et al. 2021), offline: minimise mean softmax entropy over BN affine params, then re-estimate BN stats."""
    torch.manual_seed(seed)
    m = copy.deepcopy(model).train()
    params = [p for bn in _bn_layers(m) for p in (bn.weight, bn.bias) if p is not None]
    for p in m.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    steps = 0
    for b in _batches(len(X), batch_size, epochs, seed, X.device):
        z = m(X[b])
        loss = -(F.softmax(z, 1) * F.log_softmax(z, 1)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps += 1
    for p in m.parameters():
        p.requires_grad_(True)
    _stats_pass(m, X, 1024, seed)
    used = steps * batch_size / len(X)
    return m, {"fwd_per_window": used + 1.0, "bwd_per_window": used, "n_updated": sum(p.numel() for p in params)}


def pseudo_label(model, X, tau: float, lr: float, epochs: int, batch_size: int = 256, seed: int = 0):
    """Naive self-training: fine-tune all params with CE on windows whose source max-softmax >= tau.

    Pseudo-labels come from the source model once, before any update. Fewer than one batch selected -> source copy.
    """
    torch.manual_seed(seed)
    m = copy.deepcopy(model).eval()
    conf, labels = F.softmax(_forward(m, X, 4096), 1).max(1)
    sel = torch.nonzero(conf >= tau).squeeze(1)
    n_classes = int(_forward(m, X[:2], 2).shape[1])
    info = {"selected": int(len(sel)),
            "selected_per_class": torch.bincount(labels[sel], minlength=n_classes).tolist()}
    if len(sel) < batch_size:
        return m, {**info, "selected": 0, "fwd_per_window": 1.0, "bwd_per_window": 0.0, "n_updated": 0}
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xs, ys = X[sel], labels[sel]
    steps = 0
    for b in _batches(len(sel), batch_size, epochs, seed, X.device):
        opt.zero_grad()
        F.cross_entropy(m(Xs[b]), ys[b]).backward()
        opt.step()
        steps += 1
    used = steps * batch_size / len(X)
    return m.eval(), {**info, "fwd_per_window": 1.0 + used, "bwd_per_window": used,
                      "n_updated": sum(p.numel() for p in m.parameters())}


if __name__ == "__main__":
    torch.manual_seed(0)
    centres = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])

    def blobs(n):
        y = torch.arange(3).repeat_interleave(n)
        return centres[y] + 0.5 * torch.randn(len(y), 2), y

    Xs, ys = blobs(1000)
    src = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 3))
    opt = torch.optim.Adam(src.parameters(), lr=0.01)
    for _ in range(300):
        b = torch.randint(0, len(Xs), (128,))
        opt.zero_grad()
        F.cross_entropy(src(Xs[b]), ys[b]).backward()
        opt.step()
    src.eval()
    Xt, yt = blobs(1000)
    Xt = 2 * Xt + 1  # receiver-like shift: gain and offset

    def acc(m, X=Xt):
        return (_forward(m.eval(), X, 4096).argmax(1) == yt).float().mean().item()

    def entropy(m):
        z = _forward(m.eval(), Xt, 4096)
        return -(F.softmax(z, 1) * F.log_softmax(z, 1)).sum(1).mean().item()

    before = copy.deepcopy(src.state_dict())
    affine = {f"{n}.{p}" for n, mod in src.named_modules() if isinstance(mod, nn.BatchNorm1d) for p in ("weight", "bias")}
    runs = {"adabn": lambda: adabn(src, Xt), "tent": lambda: tent(src, Xt, lr=1e-2, epochs=3),
            "pl": lambda: pseudo_label(src, Xt, tau=0.9, lr=1e-3, epochs=3)}
    out = {}
    for name, run in runs.items():
        m, info = run()
        out[name] = (m, info)
        assert all(torch.equal(before[k], v) for k, v in src.state_dict().items()), f"{name} changed the source"
        assert torch.equal(_forward(m.eval(), Xt, 4096).argmax(1), _forward(run()[0].eval(), Xt, 4096).argmax(1)), name

    m, info = out["adabn"]
    assert acc(m) > acc(src) + 0.1, (acc(src), acc(m))
    assert torch.allclose(m[0].running_mean, Xt.mean(0), atol=1e-4)
    # Xt is sorted by class: the running variance must still be the whole-set variance, not the within-class one
    assert torch.allclose(m[0].running_var, Xt.var(0), rtol=0.05), (m[0].running_var, Xt.var(0))
    assert all(torch.equal(p, before[k]) for k, p in m.named_parameters())
    assert info == {"fwd_per_window": 1.0, "bwd_per_window": 0.0, "n_updated": 0}

    m, info = out["tent"]
    assert all(torch.equal(p, before[k]) for k, p in m.named_parameters() if k not in affine)
    assert any(not torch.equal(p, before[k]) for k, p in m.named_parameters() if k in affine)
    assert entropy(m) < entropy(src), (entropy(src), entropy(m))
    assert info["n_updated"] == 2 * (2 + 16)

    m, info = out["pl"]
    conf, lab = F.softmax(_forward(src, Xt, 4096), 1).max(1)
    assert info["selected"] == int((conf >= 0.9).sum())
    assert info["selected_per_class"] == torch.bincount(lab[conf >= 0.9], minlength=3).tolist()
    m, info = pseudo_label(src, Xt, tau=1.01, lr=1e-3, epochs=1)
    assert info["selected"] == 0 and all(torch.equal(before[k], v) for k, v in m.state_dict().items())

    print({k: round(acc(v[0]), 3) for k, v in out.items()}, "source", round(acc(src), 3))
    print("adapt baselines self-check OK")
