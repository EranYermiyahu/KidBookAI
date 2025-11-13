"""
End-to-end orchestration for KidBookAI story and image generation.
"""

from .pipeline import KidBookAIOrchestrator, StoryPackage, normalize_image_outputs

__all__ = ["KidBookAIOrchestrator", "StoryPackage", "normalize_image_outputs"]

