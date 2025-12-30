from torch import nn

from nn.cnn import CNN
from nn.rnn import RNN


def get_model(name: str, input_dim: str, n_classes: int = 4) -> nn.Module:
    match name:
        case "CNN":
            return CNN(input_dim=input_dim, n_classes=n_classes)
        case "RNN":
            return RNN(input_dim=input_dim, n_classes=n_classes)
        case _:
            raise ValueError("unknown neural network architecture")
