import argparse
import os

import torch
from torch import nn
from dotenv import load_dotenv

from rfml_uav.drone_rf.utils import get_input_dim
from rfml_uav.nn.models import get_model

load_dotenv()

data_path = os.getenv("DATA_PATH")

parser = argparse.ArgumentParser(description="Export a WPD-CNN checkpoint to ONNX (softmax applied in-graph)")
parser.add_argument("--ckpt", default=os.path.join(data_path, "checkpoints/v2/checkpoint_wpd_best.pt"))
parser.add_argument("--onnx", default=os.path.join(data_path, "checkpoints/v2/checkpoint_wpd_best.onnx"))
parser.add_argument("--n-classes", type=int, default=4)
parser.add_argument("--input-dim", type=int, default=get_input_dim("Wavelets"))
parser.add_argument("--linear-tail", action="store_true")
args = parser.parse_args()

model = get_model("CNN", input_dim=args.input_dim, n_classes=args.n_classes, linear_tail=args.linear_tail)
model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
# models return logits; deployed artifacts keep returning probabilities under the name "probs"
model = nn.Sequential(model, nn.Softmax(dim=1))
model.eval()

dummy_input = torch.randn(1, 1, args.input_dim)

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
