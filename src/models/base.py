from abc import ABC, abstractmethod
from typing import Iterator
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter


class FoundationModel(ABC, nn.Module):
    """Abstract base class for foundation models."""

    def __init__(self):
        nn.Module.__init__(self)

    @abstractmethod
    def load_weights(self, checkpoint_path: str = None) -> None:
        """Load model weights from checkpoint."""
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        pass

    @abstractmethod
    def extract_embedding(self, x: Tensor) -> Tensor:
        """Extract embedding from input."""
        pass

    @abstractmethod
    def get_num_classes(self) -> int:
        """Get number of output classes."""
        pass

    @abstractmethod
    def get_parameters(self) -> Iterator[Parameter]:
        """Get model parameters."""
        pass
