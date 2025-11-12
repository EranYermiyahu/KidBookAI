"""
AI image generation package for KidBookAI.
"""

from .prompting import build_storybook_prompt
from .replicate_service import ReplicateImageGenerator

__all__ = ["build_storybook_prompt", "ReplicateImageGenerator"]


