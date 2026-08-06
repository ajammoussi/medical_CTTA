from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
from torch import Tensor
from src.models.base import FoundationModel


class CTTAAdapter(ABC):
    """Abstract base class for CTTA adapters."""

    @abstractmethod
    def setup(self, model: FoundationModel,
              source_snapshot: Optional[Dict[str, torch.Tensor]] = None,
              pretrained_snapshot: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """Setup adapter with model and optional source snapshot.

        Args:
            model: The foundation model to adapt.
            source_snapshot: Dict mapping param names to source-weight tensors
                (used by CoTTA for stochastic restore; can be None for other methods).
            pretrained_snapshot: Dict mapping param names to pretrained-weight tensors
                (used by CoTTA for teacher ensemble in cross-domain; can be None).
        """
        pass

    @abstractmethod
    def adapt_step(self, batch: Tensor, logits: Tensor) -> Dict[str, float]:
        """Perform one adaptation step."""
        pass

    @abstractmethod
    def restore_parameters(self) -> None:
        """Restore parameters to source values."""
        pass

    @abstractmethod
    def state_dict(self) -> Dict:
        """Get adapter state dict."""
        pass

    @abstractmethod
    def load_state_dict(self, state: Dict) -> None:
        """Load adapter state dict."""
        pass