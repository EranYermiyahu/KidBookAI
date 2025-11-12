"""
Convert story pages into image-ready scene descriptions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from src.common import ChatResult, CompletionCallable, call_chat_completion

from .page_splitter import StoryPage
from .profile import KidProfile


@dataclass(frozen=True)
class SceneDescription:
    """
    Represents the distilled prompt for a single illustrated scene.
    """

    page_number: int
    scene_description: str
    supporting_details: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "scene_description": self.scene_description,
            "supporting_details": self.supporting_details,
        }


class SceneDescriptionGenerator:
    """
    Converts story pages into rich scene descriptions suitable for image generation.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        completion_fn: CompletionCallable | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LITELLM_API_KEY")
        self._model = (
            model
            or os.getenv("KIDBOOKAI_SCENE_MODEL")
            or os.getenv("OPENAI_SCENE_MODEL")
            or os.getenv("LITELLM_SCENE_MODEL")
            or os.getenv("LITELLM_STORY_MODEL")
            or os.getenv("OPENAI_STORY_MODEL")
            or os.getenv("LITELLM_MODEL")
            or "gpt-4.1-mini"
        )
        self._completion_fn: CompletionCallable = completion_fn or call_chat_completion

    @property
    def model(self) -> str:
        return self._model

    def page_to_scene(
        self,
        page: StoryPage,
        *,
        profile: KidProfile,
        temperature: float = 0.4,
        max_output_tokens: int = 900,
        **response_kwargs: Any,
    ) -> SceneDescription:
        """
        Convert a single story page into a scene description for illustration guidance.
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(profile, page)

        max_tokens = max_output_tokens if max_output_tokens is not None else None
        result: ChatResult = self._completion_fn(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self._api_key,
            **response_kwargs,
        )

        payload = self._parse_scene_json(result.text)

        scene_text = payload.get("scene_description")
        supporting = payload.get("supporting_details") or ""

        if not scene_text:
            raise ValueError("Scene description response missing 'scene_description'.")

        return SceneDescription(
            page_number=page.page_number,
            scene_description=str(scene_text).strip(),
            supporting_details=str(supporting).strip(),
        )

    def _build_system_prompt(self) -> str:
        return """You are a KidBookAI scene director who transforms page-sized story beats into illustration briefs.
You collaborate with an AI art team that already understands how to preserve the child's identity.

Responsibilities:
- Distill each page into a vivid scene description that captures the location, action, emotions, and key props.
- Ensure the child hero remains the central subject and their personal traits are explicit.
- Highlight any supporting characters and how they relate physically to the child.
- Preserve all crucial narrative information from the page; do not invent new plot points.
- Mention lighting, atmosphere, or visual motifs when they support the emotion.
- Use concise, cinematic prose that an image model prompt can follow.
- Keep language inclusive, uplifting, and child-safe.

Output format:
Return valid JSON with the following fields:
{
  "scene_description": "string, 3-5 sentences describing the visual scene in rich detail.",
  "supporting_details": "string with any extra notes for wardrobe, props, or continuity. Use '' if none."
}

Do not include commentary outside the JSON.
"""

    def _build_user_prompt(self, profile: KidProfile, page: StoryPage) -> str:
        return f"""Child profile context:
{profile.summary_for_prompt()}

Page number: {page.page_number}
Page title: {page.title}
Original page text:
\"\"\"
{page.story_text}
\"\"\"

Turn this into the required JSON scene brief."""

    def _parse_scene_json(self, text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Failed to parse scene description response as JSON.") from exc

