"""Agentic layer components (Agentic_CTTA_New_Architecture.md §13.2).

Registration-trigger gotcha: every submodule used by the runners must be
imported here (HANDOFF §1), otherwise its classes never load.
"""

from src.agents.characterize import (  # noqa: F401
    Characterizer,
    DomainDescriptor,
    ImageStatistics,
    compute_image_statistics,
    descriptor_cosine,
    descriptor_l2,
    load_descriptors_json,
    write_descriptors_json,
)
from src.agents.evaluation import (  # noqa: F401
    blur_sharpness_response,
    retrieval_accuracy,
    within_between_analysis,
)

__all__ = [
    "Characterizer",
    "DomainDescriptor",
    "ImageStatistics",
    "compute_image_statistics",
    "descriptor_cosine",
    "descriptor_l2",
    "load_descriptors_json",
    "write_descriptors_json",
    "blur_sharpness_response",
    "retrieval_accuracy",
    "within_between_analysis",
]