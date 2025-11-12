"""
Service layer for producing story outlines via LiteLLM-compatible models.
"""

from __future__ import annotations

import os
from typing import Any

from src.common import ChatResult, CompletionCallable, call_chat_completion

from .profile import KidProfile
from .prompting import StoryPrompt, build_story_prompt


class StoryOutlineGenerator:
    """
    High-level helper that turns a child's profile into a complete story outline.
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
            or os.getenv("KIDBOOKAI_STORY_MODEL")
            or os.getenv("OPENAI_STORY_MODEL")
            or os.getenv("LITELLM_STORY_MODEL")
            or os.getenv("LITELLM_MODEL")
            or "gpt-4.1-mini"
        )
        self._completion_fn: CompletionCallable = completion_fn or call_chat_completion

    @property
    def model(self) -> str:
        """Return the model identifier in use."""
        return self._model

    def generate_story(
        self,
        profile: KidProfile,
        *,
        length_guidance: str | None = None,
        structure_guidance: str | None = None,
        temperature: float = 0.7,
        max_output_tokens: int = 1400,
        **response_kwargs: Any,
    ) -> str:
        """
        Invoke the configured LLM to produce the story text.
        """
        prompt_kwargs: dict[str, Any] = {}
        if length_guidance is not None:
            prompt_kwargs["length_guidance"] = length_guidance
        if structure_guidance is not None:
            prompt_kwargs["structure_guidance"] = structure_guidance

        prompt: StoryPrompt = build_story_prompt(profile, **prompt_kwargs)

        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]

        max_tokens = max_output_tokens if max_output_tokens is not None else None
        result: ChatResult = self._completion_fn(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=self._api_key,
            **response_kwargs,
        )

        if not result.text:
            raise RuntimeError("LLM response did not contain any text content.")

        return result.text

