"""CTTA Runner package.

Re-exports both CTTARunner and SequentialCTTARunner for backward compatibility.
All existing imports like `from src.evaluation.runner import CTTARunner` will continue to work.
"""

from src.evaluation.runner.base import BaseRunnerMixin
from src.evaluation.runner.single import CTTARunner, DomainResult
from src.evaluation.runner.sequential import SequentialCTTARunner

__all__ = [
    "BaseRunnerMixin",
    "CTTARunner",
    "DomainResult",
    "SequentialCTTARunner",
]
