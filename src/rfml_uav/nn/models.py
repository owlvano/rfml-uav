from torch import nn

from rfml_uav.nn.cnn import CNN
from rfml_uav.nn.rnn import RNN


def get_model(name: str, input_dim: int, n_classes: int = 4, **kwargs) -> nn.Module:
    match name:
        case "CNN":
            return CNN(input_dim=input_dim, n_classes=n_classes, **kwargs)
        case "RNN":
            return RNN(input_dim=input_dim, n_classes=n_classes, **kwargs)
        case _:
            raise ValueError("unknown neural network architecture")
