"""
Identity extraction utilities for KidBookAI illustration continuity.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Sequence

from src.common import CompletionCallable, call_chat_completion

PathLike = str | Path

logger = logging.getLogger(__name__)


_NON_PHYSICAL_PATTERN = re.compile(
    r"\b("
    r"jacket|hoodie|sweater|coat|shirt|t-shirt|tee|top|blouse|pants|jeans|shorts|skirt|dress|"
    r"outfit|clothing|attire|costume|cape|uniform|boots|shoes|sneakers|sandals|socks|"
    r"hat|beanie|cap|helmet|gloves|scarf|mask|backpack|bag|vest|overalls|glasses|goggles|"
    r"bracelet|necklace|earrings|watch|rings|belt"
    r")\b",
    re.IGNORECASE,
)


def _filter_physical_identity_notes(notes: str) -> str:
    filtered: list[str] = []
    for raw_line in notes.splitlines():
        normalized = raw_line.strip()
        if not normalized:
            continue
        normalized = normalized.lstrip("-•").strip()
        if not normalized or _NON_PHYSICAL_PATTERN.search(normalized):
            continue
        filtered.append(f"- {normalized}")
    return "\n".join(filtered)


def extract_identity_notes(
    reference_image: PathLike,
    *,
    model: str | None = None,
    api_key: str | None = None,
    completion_fn: CompletionCallable | None = None,
    max_tokens: int = 450,
    temperature: float = 0.2,
) -> str:
    """
    Use a multimodal chat model to derive bullet-point identity notes from a reference image.
    """

    completion = completion_fn or call_chat_completion
    resolved_model = (
        model
        or os.getenv("KIDBOOKAI_IDENTITY_MODEL")
        or os.getenv("OPENAI_IDENTITY_MODEL")
        or os.getenv("LITELLM_IDENTITY_MODEL")
        or "gpt-4o-mini"
    )
    resolved_api_key = api_key or os.getenv("KIDBOOKAI_IDENTITY_API_KEY") or os.getenv("OPENAI_API_KEY")

    image_payload = _normalize_image_input(reference_image)
    user_prompt = (
        "Review the child in this reference portrait and produce a concise bullet list of immutable traits. "
        "Describe only physical facial characteristics such as face shape, skin tone, eye color, eyelashes, freckles, or hair color/texture. "
        "Do not mention clothing, outfits, accessories, or props. Limit to 6-8 bullets. Use the format '- detail'."
    )

    messages: Sequence[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are an illustration continuity director. "
                "Respond only with bullet points describing the child's inherent physical facial features "
                "(face, eyes, hair, skin, freckles). "
                "Never mention clothing, outfits, accessories, or props. "
                "Do not speculate about names, backstory, or personality."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_payload}},
            ],
        },
    ]

    try:
        result = completion(
            model=resolved_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=resolved_api_key,
        )
        notes = result.text.strip()
        notes = _filter_physical_identity_notes(notes)
        return notes
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Failed to extract identity notes from reference image.")
        return ""


def _normalize_image_input(reference_image: PathLike) -> str:
    candidate = str(reference_image)
    if candidate.lower().startswith(("http://", "https://", "data:")):
        return candidate

    image_path = Path(reference_image).expanduser()
    data = image_path.read_bytes()
    mime_type, _ = mimetypes.guess_type(image_path.name)
    base64_data = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type or 'image/jpeg'};base64,{base64_data}"

