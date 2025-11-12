"""
Prompt construction utilities for KidBookAI AI image generation.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CAMERA_SHOT = (
    "medium (waist-up), eye-level, natural lens (~65mm), stable flattering perspective"
)

NEGATIVE_PROMPT = (
    "identity drift, age change, plastic skin, uncanny valley, harsh shadows, blown highlights, "
    "excessive stylization, obscured face, cluttered background, watermark, text, logo"
)


@dataclass(frozen=True)
class StorybookPrompt:
    """Container for the positive and negative prompts passed to the image model."""

    positive: str
    negative: str = NEGATIVE_PROMPT


def build_storybook_prompt(
    kid_name: str,
    scene_description: str,
    *,
    camera_shot: str | None = None,
) -> StorybookPrompt:
    """
    Build the structured prompt used to guide the image generation model.

    Parameters
    ----------
    kid_name:
        Name of the child that will appear in the illustration.
    scene_description:
        Narrative description of the scene that should be rendered.
    camera_shot:
        Optional camera guidance (e.g., "close-up portrait (shoulders-up)"). Falls back
        to a balanced default that works for most storybook scenes.
    """
    if not kid_name or not kid_name.strip():
        raise ValueError("kid_name must be a non-empty string.")

    if not scene_description or not scene_description.strip():
        raise ValueError("scene_description must be a non-empty string.")

    shot = camera_shot.strip() if camera_shot else DEFAULT_CAMERA_SHOT

    positive_prompt = f"""TASK
Edit the input photo(s) to create a high-definition storybook illustration of {kid_name}, while strictly preserving the child's facial identity and age.

IDENTITY LOCK (do not change)
- Keep the same face shape, eye shape, nose, mouth, skin tone, hair color/style, and overall proportions as in the input image(s).
- Maintain age and ethnicity exactly. Do not beautify, slim, or exaggerate features. No caricature.
- Preserve the natural expression and personality cues from the reference (joy, curiosity, calm).

SCENE (apply to background, wardrobe, pose as needed - do not alter the face)
- {scene_description}
- Child remains the clear focal point; background supports the narrative without stealing focus.

ART DIRECTION - Pixar/Disney-like 3D cinematic realism
- Smooth but natural skin (not plastic), clean materials, stylized-real anatomy.
- Vivid, balanced color palette; storybook ambience; gentle bokeh (subtle depth-of-field).
- Camera: {shot}.

LIGHTING & MOOD
- Soft, radiant key light with warm rim light that creates a magical glow; believable soft shadowing.
- Wholesome, uplifting, imaginative mood; joyful and comforting.

ENVIRONMENT & WARDROBE
- Storybook background matching the scene (enchanted forest / cozy lantern-lit town / futuristic city / starry sky).
- Wardrobe consistent with the scene (e.g., small superhero cape, explorer outfit); no real-world logos or text.

RENDERING QUALITY
- Ultra-sharp, print-ready detail; premium filmic contrast; clean shading and reflections; 8K-ready upscale look.
- Composition keeps the child's face readable and instantly recognizable."""

    return StorybookPrompt(positive=positive_prompt)


