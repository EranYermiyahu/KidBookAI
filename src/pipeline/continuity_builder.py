"""
Helpers for assembling illustration continuity configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.common import CompletionCallable
from src.story_generation import KidProfile

from .continuity import IllustrationContinuityConfig
from .identity import extract_identity_notes

PathLike = str | Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContinuityAssembly:
    """
    Result container produced by :func:`build_illustration_continuity_config`.
    """

    config: IllustrationContinuityConfig
    identity_notes: str | None
    auto_identity_notes: str | None
    supporting_character_notes: Mapping[str, str]
    static_continuity_notes: tuple[str, ...]


def build_illustration_continuity_config(
    *,
    profile: KidProfile,
    reference_image: PathLike,
    automatic_identity_notes: bool = True,
    identity_note_overrides: Sequence[str] | str | None = None,
    supporting_character_overrides: Mapping[str, str] | None = None,
    static_continuity_overrides: Sequence[str] | None = None,
    reference_history_size: int | None = None,
    reference_history_parameter: str | None = None,
    promote_latest_reference: bool | None = None,
    identity_model: str | None = None,
    identity_api_key: str | None = None,
    completion_fn: CompletionCallable | None = None,
) -> ContinuityAssembly:
    """
    Combine reference image insights, profile cues, and overrides into a continuity config.
    """

    resolved_ref_image = reference_image
    base_defaults = IllustrationContinuityConfig()
    ref_history_size = (
        reference_history_size
        if reference_history_size is not None
        else base_defaults.reference_history_size
    )
    ref_history_size = max(0, ref_history_size)
    ref_history_parameter = (
        reference_history_parameter
        if reference_history_parameter is not None
        else base_defaults.reference_history_parameter
    )
    use_promote_latest = (
        promote_latest_reference
        if promote_latest_reference is not None
        else base_defaults.promote_latest_reference
    )

    manual_identity_lines = _collect_note_lines(identity_note_overrides)
    profile_identity_lines = _collect_note_lines(profile.identity_traits)

    auto_identity_notes = ""
    if automatic_identity_notes:
        try:
            auto_identity_notes = extract_identity_notes(
                resolved_ref_image,
                model=identity_model,
                api_key=identity_api_key,
                completion_fn=completion_fn,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Identity extraction failed; proceeding without auto-notes.")
            auto_identity_notes = ""
    auto_identity_lines = _collect_note_lines(auto_identity_notes)

    identity_lines = _deduplicate(
        manual_identity_lines + profile_identity_lines + auto_identity_lines
    )
    identity_notes = (
        "\n".join(f"- {line}" for line in identity_lines) if identity_lines else None
    )

    supporting_notes = dict(profile.supporting_characters)
    if supporting_character_overrides:
        for name, details in supporting_character_overrides.items():
            name_text = str(name).strip()
            details_text = str(details).strip()
            if name_text and details_text:
                supporting_notes[name_text] = details_text

    static_notes: list[str] = list(profile.continuity_notes)
    if static_continuity_overrides:
        static_notes.extend(_collect_note_lines(static_continuity_overrides))
    static_notes = _deduplicate(static_notes)

    should_carry_history = ref_history_size > 0
    resolved_history_parameter = (
        ref_history_parameter if should_carry_history else None
    )
    resolved_promote_latest = use_promote_latest if should_carry_history else False

    config = IllustrationContinuityConfig(
        identity_notes=identity_notes,
        supporting_character_notes=supporting_notes,
        static_continuity_notes=tuple(static_notes),
        carry_reference_image_forward=should_carry_history,
        reference_history_size=ref_history_size,
        reference_history_parameter=resolved_history_parameter,
        promote_latest_reference=resolved_promote_latest,
        propagate_supporting_details=True,
        supporting_details_history=base_defaults.supporting_details_history,
        locked_seed=None,
        auto_seed=True,
    )

    return ContinuityAssembly(
        config=config,
        identity_notes=identity_notes,
        auto_identity_notes=auto_identity_notes or None,
        supporting_character_notes=supporting_notes,
        static_continuity_notes=tuple(static_notes),
    )


def _collect_note_lines(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        source = value
    elif isinstance(value, Mapping):
        source = "\n".join(str(item) for item in value.values())
    else:
        source_parts: list[str] = []
        for item in value:
            if item is None:
                continue
            source_parts.append(str(item))
        source = "\n".join(source_parts)

    lines: list[str] = []
    for raw in source.replace("\r", "\n").split("\n"):
        cleaned = raw.strip(" \t-•—")
        if cleaned:
            lines.append(cleaned)
    return lines


def _deduplicate(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            ordered.append(item)
            seen.add(key)
    return ordered

