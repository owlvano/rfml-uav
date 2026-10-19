from typing import Literal

import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def _splitter(mode: str, n_splits: int, seed: int):
    match mode:
        case "recording":
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        case "chunk":
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        case _:
            raise ValueError(f"unknown split mode: {mode}")


def make_splits(
    y,
    groups,
    mode: Literal["chunk", "recording"],
    n_folds: int = 5,
    val_frac: float = 0.2,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Outer K-fold test folds, each with a validation set carved from its train part.

    mode="recording": every group (recording, e.g. f"{bui}_{n}") lands in exactly one of train/val/test.
    mode="chunk":     legacy paper-1 protocol, chunks split independently of recording (leaks by design).
    Returns [(train_idx, val_idx, test_idx), ...] of length n_folds.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    classes = np.unique(y)
    inner_splits = round(1 / val_frac)
    g = groups if mode == "recording" else None  # StratifiedKFold warns on groups
    folds = []
    for train_val, test in _splitter(mode, n_folds, seed).split(y, y, g):
        assert set(np.unique(y[test])) == set(classes), "a test fold is missing a class"
        g_tv = None if g is None else g[train_val]
        tr, va = next(_splitter(mode, inner_splits, seed).split(y[train_val], y[train_val], g_tv))
        folds.append((train_val[tr], train_val[va], test))
    return folds



def undersample(idx, y, seed: int) -> np.ndarray:
    """Rows `idx` with every class undersampled to the smallest (paper-3 source protocol), sorted.

    y is indexed by idx (y[idx] are the labels of those rows).
    """
    idx = np.asarray(idx)
    rus = RandomUnderSampler(sampling_strategy="not minority", random_state=seed)
    rus.fit_resample(idx.reshape(-1, 1), np.asarray(y)[idx])
    return np.sort(idx[rus.sample_indices_])


if __name__ == "__main__":
    # 4 classes x 21 recordings x 30 chunks
    y = np.repeat(np.arange(4), 21 * 30)
    groups = np.array([f"{c}_{r}" for c in range(4) for r in range(21) for _ in range(30)])

    for mode in ("recording", "chunk"):
        folds = make_splits(y, groups, mode, seed=3)
        assert len(folds) == 5
        all_test = np.concatenate([te for _, _, te in folds])
        assert np.array_equal(np.sort(all_test), np.arange(len(y)))  # test folds partition the data
        for tr, va, te in folds:
            assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
            if mode == "recording":
                gtr, gva, gte = set(groups[tr]), set(groups[va]), set(groups[te])
                assert not (gtr & gva or gtr & gte or gva & gte)
        again = make_splits(y, groups, mode, seed=3)
        assert all(all(np.array_equal(a, b) for a, b in zip(f1, f2)) for f1, f2 in zip(folds, again))

    # chunk mode really does share recordings between train and test
    tr, _, te = make_splits(y, groups, "chunk", seed=0)[0]
    assert set(groups[tr]) & set(groups[te])
    # undersample: balanced, deterministic per seed, a subset of idx
    yy = np.repeat(np.arange(3), [50, 80, 120])
    sub = undersample(np.arange(len(yy)), yy, seed=1)
    assert np.array_equal(np.bincount(yy[sub]), [50, 50, 50])
    assert np.array_equal(sub, undersample(np.arange(len(yy)), yy, seed=1))
    print("splits self-check OK")
