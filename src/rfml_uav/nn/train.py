import torch
import torch.nn as nn
from sklearn.metrics import f1_score


def predict_logits(model: nn.Module, X: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + batch_size]) for i in range(0, len(X), batch_size)])


def train_best_val(model: nn.Module, X: torch.Tensor, y: torch.Tensor, tr, va,
                   epochs: int = 50, lr: float = 0.01, batch_size: int = 256, seed: int = 0):
    """Adam + cross-entropy for `epochs`; returns (model at best val macro-F1 epoch, that F1, that epoch).

    X, y live on the training device; tr / va index into them. Call torch.manual_seed(seed) before
    building the model if initialisation must be reproducible too.
    """
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    tr_t, y_va = torch.as_tensor(tr, device=X.device), y[va].cpu().numpy()
    best = (-1.0, 0, None)
    for epoch in range(epochs):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), device=X.device)]
        for i in range(0, len(perm) - batch_size + 1, batch_size):  # drop last partial batch (BatchNorm)
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(X[b]), y[b]).backward()
            opt.step()
        val_f1 = f1_score(y_va, predict_logits(model, X[va]).argmax(1).cpu().numpy(), average="macro")
        if val_f1 > best[0]:
            best = (val_f1, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    return model, best[0], best[1]
