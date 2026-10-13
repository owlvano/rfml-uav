import argparse
import os

import torch
from torch import nn
from dotenv import load_dotenv

from rfml_uav.drone_rf.utils import get_input_dim
from rfml_uav.nn.models import get_model

load_dotenv()

data_path = os.getenv("DATA_PATH")

N_CLASSES = 4
INPUT_DIM = get_input_dim("Wavelets")

parser = argparse.ArgumentParser(description="Export a WPD-CNN checkpoint to ONNX (softmax applied in-graph)")
parser.add_argument("--ckpt", default=os.path.join(data_path, "checkpoints/v2/checkpoint_wpd_best.pt"))
parser.add_argument("--onnx", default=os.path.join(data_path, "checkpoints/v2/checkpoint_wpd_best.onnx"))
args = parser.parse_args()

model = get_model("CNN", input_dim=INPUT_DIM, n_classes=N_CLASSES)
model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
# models return logits; deployed artifacts keep returning probabilities under the name "probs"
model = nn.Sequential(model, nn.Softmax(dim=1))
model.eval()

dummy_input = torch.randn(1, 1, INPUT_DIM)

torch.onnx.export(
    model,
    dummy_input,
    args.onnx,
    input_names=["input"],
    output_names=["probs"],
    dynamic_axes={"input": {0: "batch"}, "probs": {0: "batch"}},
    opset_version=17,
    dynamo=False,
)

print(f"Exported: {args.onnx}")
