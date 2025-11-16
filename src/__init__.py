"""
KidBookAI package exposing story generation, pipeline, and PDF tooling.
"""

from .pdf_generation import StorybookPDFBuilder
from .pipeline import (
    ContinuityAssembly,
    IllustrationContinuityConfig,
    KidBookAIOrchestrator,
    StoryPackage,
    build_illustration_continuity_config,
)

__all__ = [
    "ContinuityAssembly",
    "IllustrationContinuityConfig",
    "build_illustration_continuity_config",
    "KidBookAIOrchestrator",
    "StoryPackage",
    "StorybookPDFBuilder",
]

