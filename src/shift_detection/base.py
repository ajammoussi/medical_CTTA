from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
from torch import Tensor


class ShiftDetector(ABC):
    """Abstract base class for shift detection."""

    @abstractmethod
    def update(self, batch: Tensor, logits: Tensor,
               embeddings: Optional[Tensor] = None) -> None:
        """Update detector with new batch."""
        pass

    @abstractmethod
    def detect(self) -> bool:
        """Detect if shift occurred."""
        pass

    @abstractmethod
    def get_signals(self) -> Dict[str, float]:
        """Get current signal values."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset detector state."""
        pass