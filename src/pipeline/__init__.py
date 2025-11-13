"""
End-to-end orchestration for KidBookAI story and image generation.
"""

from .continuity import IllustrationContinuityConfig
from .pipeline import KidBookAIOrchestrator, StoryPackage, normalize_image_outputs

__all__ = [
    "IllustrationContinuityConfig",
    "KidBookAIOrchestrator",
    "StoryPackage",
    "normalize_image_outputs",
]

