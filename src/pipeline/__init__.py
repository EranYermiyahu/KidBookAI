"""
End-to-end orchestration for KidBookAI story and image generation.
"""

from .continuity import IllustrationContinuityConfig
from .continuity_builder import (
    ContinuityAssembly,
    build_illustration_continuity_config,
)
from .pipeline import KidBookAIOrchestrator, StoryPackage, normalize_image_outputs

__all__ = [
    "IllustrationContinuityConfig",
    "ContinuityAssembly",
    "build_illustration_continuity_config",
    "KidBookAIOrchestrator",
    "StoryPackage",
    "normalize_image_outputs",
]

