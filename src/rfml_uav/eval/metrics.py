import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def classification_report_dict(y_true, y_pred, probs, class_names) -> dict:
    """Scalar + per-class metrics, confusion matrix, and max-softmax confidence split by correctness."""
    y_true, y_pred, probs = np.asarray(y_true), np.asarray(y_pred), np.asarray(probs)
    labels = list(range(len(class_names)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    correct = y_true == y_pred
    conf = probs.max(axis=1)
    report = {
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "conf_correct": conf[correct],
        "conf_wrong": conf[~correct],
    }
    for i, name in enumerate(class_names):
        report[f"{name}_precision"], report[f"{name}_recall"], report[f"{name}_f1"] = map(float, (p[i], r[i], f[i]))
    return report


if __name__ == "__main__":
    names = ["Background", "Bebop", "AR", "Phantom"]
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    probs = np.eye(4)[y] * 0.9 + 0.025
    rep = classification_report_dict(y, y, probs, names)
    assert rep["accuracy"] == 1.0 and rep["macro_f1"] == 1.0
    assert len(rep["conf_correct"]) == 8 and len(rep["conf_wrong"]) == 0
    assert np.array_equal(rep["confusion_matrix"], np.diag([2, 2, 2, 2]))

    y_pred = y.copy(); y_pred[0] = 1  # one Background predicted as Bebop
    rep = classification_report_dict(y, y_pred, probs, names)
    assert rep["accuracy"] == 7 / 8 and rep["Background_recall"] == 0.5 and rep["Bebop_precision"] == 2 / 3
    assert len(rep["conf_wrong"]) == 1
    print("metrics self-check OK")
