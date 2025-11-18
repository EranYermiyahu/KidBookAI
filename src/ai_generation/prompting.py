"""
Prompt construction utilities for KidBookAI AI image generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

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
    favorite_theme: str | Sequence[str] | None = None,
    outfit_description: str | None = None,
    facial_expression: str | Sequence[str] | None = None,
    pose_description: str | Sequence[str] | None = None,
    identity_traits: str | Sequence[str] | None = None,
    continuity_notes: Sequence[str] | None = None,
    supporting_cast_notes: Sequence[str] | Mapping[str, str] | None = None,
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
    favorite_theme:
        Optional thematic cues (e.g., "Superhero adventures") that should inform palette,
        mood, and wardrobe motifs.
    outfit_description:
        Optional wardrobe guidance tailored to this page that should inform the illustration.
    facial_expression:
        Optional description of the child's facial expression for this scene.
    pose_description:
        Optional guidance on the child's pose or body language for this scene.
    """
    if not kid_name or not kid_name.strip():
        raise ValueError("kid_name must be a non-empty string.")

    if not scene_description or not scene_description.strip():
        raise ValueError("scene_description must be a non-empty string.")

    shot = camera_shot.strip() if camera_shot else DEFAULT_CAMERA_SHOT

    theme_clause = ""
    if favorite_theme:
        theme_clause = f"\n- Let the vibe reflect the child's favorite theme: {favorite_theme}"

    positive_prompt = f"""TASK
Edit the input photo(s) to create a high-definition storybook illustration of {kid_name}, while strictly preserving the child's facial identity and age.

IDENTITY LOCK (do not change)
- Keep the same face shape, eye shape, nose, mouth, skin tone, hair color/style, and overall proportions as in the input image(s).
- Maintain age and ethnicity exactly. Do not beautify, slim, or exaggerate features. No caricature.
- Adapt the child's facial expression to fit the scene description while preserving facial likeness; the reference expression is optional.

REFERENCE IMAGE USAGE
- Use the input photo only to match the child's facial structure, skin tone, eye color, and hair details.
- Ignore outfits, clothing, shoes, or accessories from the reference image; treat them purely as placeholders.
- Focus on faithful facial and physical traits. Clothing, facial expression, and pose should all come from the story context, not the reference image.

SCENE (apply to background, wardrobe, pose as needed - do not alter the face)
- {scene_description}
- Child remains the clear focal point; background supports the narrative without stealing focus.

ART DIRECTION - Pixar/Disney-like 3D cinematic realism
- Smooth but natural skin (not plastic), clean materials, stylized-real anatomy.
- Vivid, balanced color palette; storybook ambience; gentle bokeh (subtle depth-of-field).
- Camera: {shot}.{theme_clause}

LIGHTING & MOOD
- Soft, radiant key light with warm rim light that creates a magical glow; believable soft shadowing.
- Wholesome, uplifting, imaginative mood; joyful and comforting.

ENVIRONMENT & WARDROBE
- Storybook background matching the scene (enchanted forest / cozy lantern-lit town / futuristic city / starry sky).
- Wardrobe consistent with the scene (e.g., small superhero cape, explorer outfit); do not copy the outfit from the reference photo. Ground clothing choices in the provided outfit guidance. No real-world logos or text.

RENDERING QUALITY
- Must deliver a high-definition, high-quality illustration suitable for large-format print.
- Ultra-sharp, print-ready detail; premium filmic contrast; clean shading and reflections; 8K-ready upscale look.
- Composition keeps the child's face readable and instantly recognizable."""

    extra_sections: list[str] = []

    theme_lines = _normalize_note_input(favorite_theme)
    if theme_lines:
        theme_notes = list(theme_lines)
        theme_notes.append(
            "Let this theme drive the overall mood, palette, wardrobe motifs, and props."
        )
        extra_sections.append(_format_bullet_section("STYLE & ATMOSPHERE", theme_notes))

    outfit_lines = _normalize_note_input(outfit_description)
    if outfit_lines:
        outfit_notes = list(outfit_lines)
        outfit_notes.append(
            "Derive clothing from the story text, scene setting, and child profile; never copy garments from the reference photo."
        )
        extra_sections.append(
            _format_bullet_section("OUTFIT DESIGN (scene-specific)", outfit_notes)
        )

    expression_lines = _normalize_note_input(facial_expression)
    if expression_lines:
        expression_notes = list(expression_lines)
        expression_notes.append(
            "Keep the expression natural, age-appropriate, and aligned with the story emotion and favorite theme—not the reference photo."
        )
        extra_sections.append(
            _format_bullet_section("FACIAL EXPRESSION", expression_notes)
        )

    pose_lines = _normalize_note_input(pose_description)
    if pose_lines:
        pose_notes = list(pose_lines)
        pose_notes.append(
            "Ensure the pose feels comfortable, safe, and supports the narrative action and favorite theme described in the scene; ignore the reference photo pose."
        )
        extra_sections.append(_format_bullet_section("POSE GUIDANCE", pose_notes))

    identity_lines = _normalize_note_input(identity_traits)
    if identity_lines:
        extra_sections.append(_format_bullet_section("IDENTITY SNAPSHOT", identity_lines))

    supporting_lines = _normalize_note_input(supporting_cast_notes)
    if supporting_lines:
        extra_sections.append(
            _format_bullet_section("SUPPORTING CAST CONTINUITY", supporting_lines)
        )

    continuity_lines = _normalize_note_input(continuity_notes)
    if continuity_lines:
        extra_sections.append(_format_bullet_section("CONTINUITY NOTES", continuity_lines))

    if extra_sections:
        positive_prompt = positive_prompt + "\n\n" + "\n\n".join(extra_sections)

    return StorybookPrompt(positive=positive_prompt)


def _normalize_note_input(
    value: str | Sequence[str] | Mapping[str, str] | None,
) -> list[str]:
    if value is None:
        return []

    if isinstance(value, Mapping):
        items = [f"{key}: {details}" for key, details in value.items()]
    elif isinstance(value, str):
        items = [value]
    else:
        items = [str(item) for item in value]

    lines: list[str] = []
    for item in items:
        for raw in item.replace("\r", "\n").split("\n"):
            cleaned = raw.strip(" \t-•—")
            if cleaned:
                lines.append(cleaned)
    return lines


def _format_bullet_section(title: str, lines: Sequence[str]) -> str:
    bullet_block = "\n".join(f"- {line}" for line in lines if line.strip())
    return f"{title}\n{bullet_block}"


