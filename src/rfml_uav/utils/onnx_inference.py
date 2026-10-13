import argparse
import os

import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv

from rfml_uav.drone_rf.consts import BUI_dict

CLASS_NAMES = list(BUI_dict.keys())  # index order matches get_class_label: Background, Bebop, AR, Phantom

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH")

FEATURE_FILES = {"psd": "PSD", "stft": "STFT", "wpd": "Wavelets"}


def main():
    parser = argparse.ArgumentParser(description="Run inference with an exported ONNX checkpoint on real balanced samples")
    parser.add_argument("--onnx", required=True, help="path to .onnx checkpoint")
    parser.add_argument("--feature", choices=FEATURE_FILES.keys(), required=True, help="feature type the model was trained on")
    parser.add_argument("--n-samples", type=int, default=20, help="number of random samples to test")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    npz_path = os.path.join(DATA_PATH, "balanced", f"{FEATURE_FILES[args.feature]}.npz")
    raw = np.load(npz_path)
    X, y = raw["X"], raw["y"].astype(np.int64)

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(X), size=min(args.n_samples, len(X)), replace=False)
    batch_X = X[idx][:, None, :].astype(np.float32)  # (N, 1, F)
    batch_y = y[idx]

    sess = ort.InferenceSession(args.onnx)
    probs = sess.run(None, {"input": batch_X})[0]
    preds = probs.argmax(axis=1)

    correct = 0
    for true, pred, p in zip(batch_y, preds, probs):
        ok = true == pred
        correct += ok
        print(f"true={CLASS_NAMES[true]:<10} pred={CLASS_NAMES[pred]:<10} conf={p[pred]:.3f}  {'OK' if ok else 'MISS'}")

    print(f"\nAccuracy: {correct}/{len(batch_y)} = {100.0 * correct / len(batch_y):.1f}%")


if __name__ == "__main__":
    main()
