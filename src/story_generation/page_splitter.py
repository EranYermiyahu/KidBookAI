"""
Utilities for splitting a full story into illustrated pages.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from src.common import ChatResult, CompletionCallable, call_chat_completion

from .profile import KidProfile

DEFAULT_PAGE_COUNT_RANGE = (12, 18)


@dataclass(frozen=True)
class StoryPage:
    """
    Represents a single illustrated page derived from the full story text.
    """

    page_number: int
    title: str
    story_text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "title": self.title,
            "story_text": self.story_text,
        }


class StoryPageSplitter:
    """
    Splits a story into self-contained pages suitable for illustration.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        completion_fn: CompletionCallable | None = None,
        target_page_range: tuple[int, int] = DEFAULT_PAGE_COUNT_RANGE,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LITELLM_API_KEY")
        self._model = (
            model
            or os.getenv("KIDBOOKAI_PAGE_MODEL")
            or os.getenv("OPENAI_PAGE_SPLITTER_MODEL")
            or os.getenv("LITELLM_PAGE_MODEL")
            or os.getenv("LITELLM_STORY_MODEL")
            or os.getenv("OPENAI_STORY_MODEL")
            or os.getenv("LITELLM_MODEL")
            or "gpt-4.1-mini"
        )
        self._completion_fn: CompletionCallable = completion_fn or call_chat_completion
        self._target_page_range = target_page_range

    @property
    def model(self) -> str:
        return self._model

    def split_story(
        self,
        story_markdown: str,
        *,
        profile: KidProfile,
        desired_pages: int | None = None,
        temperature: float = 0.3,
        max_output_tokens: int = 2500,
        **response_kwargs: Any,
    ) -> list[StoryPage]:
        """
        Split the story into 12-18 illustrated pages using the LLM as a judge.
        """
        if not story_markdown.strip():
            raise ValueError("Story text must be a non-empty string.")

        desired_count = desired_pages or sum(self._target_page_range) // 2
        lower, upper = self._target_page_range
        if not lower <= desired_count <= upper:
            raise ValueError(
                f"desired_pages must fall between {lower} and {upper}, received {desired_count}."
            )

        system_prompt = self._build_system_prompt(lower, upper)
        user_prompt = self._build_user_prompt(profile, story_markdown, desired_count)

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

        raw_text = result.text
        pages_data = self._parse_pages_json(raw_text)
        pages = self._convert_to_pages(pages_data)
        self._validate_page_sequence(pages, lower, upper)
        return pages

    def _build_system_prompt(self, lower: int, upper: int) -> str:
        return f"""You are an award-winning children's book editor who structures stories into illustrated pages.
You receive a complete KidBookAI tale and decide how to split it so each page supports a vivid illustration while preserving the full narrative.

Responsibilities:
- Maintain chronological order and keep the child's identity, traits, and relationships explicit.
- Every page must stand alone with enough context for an illustrator to depict the scene.
- Preserve essential plot beats, emotions, and descriptive details from the original story.
- Aim for {lower}-{upper} total pages; do not exceed the upper bound.
- Provide each page with a short, playful title highlighting the key moment.
- Keep text balanced so the story flows naturally when read page by page.
- Use child-safe, inclusive language only.

Output format:
Respond with valid JSON matching this schema:
{{
  "pages": [
    {{
      "page_number": 1,
      "title": "string, 3-7 words capturing the moment",
      "story_text": "string, 2-4 sentences for that page"
    }},
    ...
  ]
}}

Do not include commentary outside the JSON."""

    def _build_user_prompt(
        self, profile: KidProfile, story_markdown: str, desired_pages: int
    ) -> str:
        return f"""Child profile:
{profile.summary_for_prompt()}

Desired total pages: {desired_pages}

Full story to split:
\"\"\"markdown
{story_markdown}
\"\"\"

Split the story according to the instructions."""

    def _parse_pages_json(self, raw_text: str) -> Sequence[dict[str, Any]]:
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError("Failed to parse page split response as JSON.") from exc

        pages = parsed.get("pages")
        if not isinstance(pages, list):
            raise ValueError("Page split JSON must contain a 'pages' list.")

        return pages

    def _convert_to_pages(self, pages_data: Iterable[dict[str, Any]]) -> list[StoryPage]:
        pages: list[StoryPage] = []
        for item in pages_data:
            try:
                number = int(item["page_number"])
                title = str(item["title"]).strip()
                story_text = str(item["story_text"]).strip()
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid page payload: {item}") from exc

            if not title or not story_text:
                raise ValueError(f"Page {number} is missing title or story_text content.")

            pages.append(StoryPage(page_number=number, title=title, story_text=story_text))
        return pages

    def _validate_page_sequence(self, pages: Sequence[StoryPage], lower: int, upper: int) -> None:
        if not lower <= len(pages) <= upper:
            raise ValueError(
                f"Expected between {lower} and {upper} pages, received {len(pages)}."
            )

        for expected, page in enumerate(pages, start=1):
            if page.page_number != expected:
                raise ValueError("Page numbers must be sequential starting from 1.")

