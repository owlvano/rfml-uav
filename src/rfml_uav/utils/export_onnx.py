import os

import torch
from dotenv import load_dotenv

from rfml_uav.drone_rf.utils import get_input_dim
from rfml_uav.nn.models import get_model

load_dotenv()

data_path = os.getenv("DATA_PATH")
ckpt_path = os.path.join(data_path, "checkpoints/v2/checkpoint_wpd_best.pt")
onnx_path = os.path.join(data_path, "checkpoints/v2/checkpoint_wpd_best.onnx")

N_CLASSES = 4
INPUT_DIM = get_input_dim("Wavelets")

model = get_model("CNN", input_dim=INPUT_DIM, n_classes=N_CLASSES)
model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 1, INPUT_DIM)

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["probs"],
    dynamic_axes={"input": {0: "batch"}, "probs": {0: "batch"}},
    opset_version=17,
    dynamo=False,
)

print(f"Exported: {onnx_path}")
