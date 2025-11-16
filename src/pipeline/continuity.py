"""
Continuity helpers to keep KidBookAI illustrations consistent across pages.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Deque, Mapping, Sequence

if TYPE_CHECKING:
    from src.story_generation.scene_builder import SceneDescription
    from src.story_generation.profile import KidProfile

from src.story_generation.profile import KidProfile

PathLike = str | Path


@dataclass(frozen=True)
class IllustrationContinuityConfig:
    """
    Configuration knobs that control illustration continuity behaviour.

    Attributes
    ----------
    identity_notes:
        Optional bullet-style description of the child's defining traits
        (hair, skin tone, facial features). Included in every prompt.
    supporting_character_notes:
        Mapping of supporting character names to descriptions that should
        remain stable across illustrations.
    static_continuity_notes:
        Additional always-on continuity reminders injected into the prompt
        (e.g., wardrobe must stay the same).
    carry_reference_image_forward:
        Whether to retain previously generated illustration URLs for reuse. Disabled by
        default to avoid unintended wardrobe locking.
    reference_history_size:
        Maximum number of prior illustration URLs to reuse as references when carrying
        history forward.
    reference_history_parameter:
        Optional name of the model input parameter that should receive the
        reference history list (e.g., ``reference_images``). When ``None``,
        the history will not be forwarded automatically.
    promote_latest_reference:
        When True, prefer the most recent generated illustration as the primary
        reference image for the next request. The original photo is still kept
        in the additional references list.
    propagate_supporting_details:
        When True, the `supporting_details` emitted by the scene generator
        are threaded into future prompts as continuity reminders.
    supporting_details_history:
        Maximum number of previous supporting detail snippets to keep.
    locked_seed:
        Optional deterministic seed forwarded to the image model if it supports
        a `seed` parameter. User-specified `image_kwargs` take precedence.
    auto_seed:
        When True (default), derive a deterministic seed from the child profile
        and reference image. Ignored if ``locked_seed`` is provided.
    vary_seed_per_page:
        When True, increment the derived seed per page to create a deterministic
        sequence. When False (default), reuse a single seed for the full story.
    """

    identity_notes: str | None = None
    supporting_character_notes: Mapping[str, str] = field(default_factory=dict)
    static_continuity_notes: Sequence[str] = field(default_factory=tuple)
    carry_reference_image_forward: bool = False
    reference_history_size: int = 0
    reference_history_parameter: str | None = None
    promote_latest_reference: bool = False
    propagate_supporting_details: bool = True
    supporting_details_history: int = 2
    locked_seed: int | None = None
    auto_seed: bool = True
    vary_seed_per_page: bool = False


@dataclass(frozen=True)
class GenerationContinuityDirectives:
    """
    Fully-resolved continuity directives for a single illustration request.
    """

    primary_reference_image: PathLike
    additional_reference_images: tuple[PathLike, ...] = ()
    identity_notes: str | None = None
    continuity_notes: tuple[str, ...] = ()
    supporting_cast_notes: tuple[str, ...] = ()
    reference_history_parameter: str | None = None
    model_overrides: Mapping[str, Any] = field(default_factory=dict)

    def as_prompt_payload(self) -> dict[str, Sequence[str] | str | None]:
        """
        Convenience helper mainly used for testing.
        """
        return {
            "identity_notes": self.identity_notes,
            "continuity_notes": self.continuity_notes,
            "supporting_cast_notes": self.supporting_cast_notes,
        }


class IllustrationContinuityState:
    """
    Tracks continuity data while the pipeline renders sequential illustrations.
    """

    _SEED_MOD = 2_147_483_647

    def __init__(
        self,
        *,
        profile: KidProfile,
        base_reference_image: PathLike,
        config: IllustrationContinuityConfig | None = None,
    ) -> None:
        self._config = config or IllustrationContinuityConfig()
        self._profile = profile
        self._base_reference_image: PathLike = str(base_reference_image)
        history_size = max(0, self._config.reference_history_size)
        supporting_history_size = max(0, self._config.supporting_details_history)
        self._reference_history: Deque[str] = deque(maxlen=history_size or None)
        self._supporting_history: Deque[str] = deque(maxlen=supporting_history_size)
        self._identity_notes = self._config.identity_notes or self._format_identity_notes(
            profile
        )
        self._seed_mod = self._SEED_MOD
        self._base_seed = None
        if self._config.locked_seed is not None:
            self._base_seed = int(self._config.locked_seed) % self._seed_mod or 1
        elif self._config.auto_seed:
            self._base_seed = self._derive_consistent_seed(
                profile=profile,
                reference_image=base_reference_image,
            )

    def build_directives(
        self,
        *,
        scene: "SceneDescription",
        page_number: int,
    ) -> GenerationContinuityDirectives:
        identity_notes = self._sanitize(self._identity_notes)
        continuity_notes: list[str] = []

        static_notes = [
            self._sanitize(note) for note in self._config.static_continuity_notes
        ]
        continuity_notes.extend(filter(None, static_notes))

        if self._config.propagate_supporting_details:
            continuity_notes.extend(self._supporting_history)

        current_supporting = self._summarize_supporting_details(scene.supporting_details)
        for note in current_supporting:
            if note and note not in continuity_notes:
                continuity_notes.append(note)

        supporting_cast_notes = [
            f"{name}: {details}"
            for name, details in self._config.supporting_character_notes.items()
            if self._sanitize(name) and self._sanitize(details)
        ]

        primary_reference: PathLike = self._base_reference_image
        additional_refs: tuple[PathLike, ...] = ()

        history_list: list[str] = []
        if self._config.carry_reference_image_forward:
            history_list = list(self._reference_history)
            if self._config.promote_latest_reference and history_list:
                primary_reference = history_list[0]
                history_list = history_list[1:]
                if self._base_reference_image not in history_list:
                    history_list.append(self._base_reference_image)

            if history_list:
                additional_refs = tuple(history_list)

        reference_history_param = (
            self._config.reference_history_parameter
            if self._config.carry_reference_image_forward
            else None
        )

        model_overrides: dict[str, Any] = {}
        seed = self.seed_for_page(page_number)
        if seed is not None:
            model_overrides["seed"] = seed

        return GenerationContinuityDirectives(
            primary_reference_image=primary_reference,
            additional_reference_images=additional_refs,
            identity_notes=identity_notes,
            continuity_notes=tuple(filter(None, continuity_notes)),
            supporting_cast_notes=tuple(supporting_cast_notes),
            reference_history_parameter=reference_history_param,
            model_overrides=model_overrides,
        )

    def record_generation(
        self,
        *,
        scene: "SceneDescription",
        image_outputs: Sequence[str],
    ) -> None:
        """
        Update the continuity buffers with fresh supporting notes and references.
        """
        if self._config.propagate_supporting_details:
            notes = self._summarize_supporting_details(scene.supporting_details)
            for note in notes:
                if note and note not in self._supporting_history:
                    self._supporting_history.appendleft(note)

        if not self._config.carry_reference_image_forward:
            return

        for candidate in image_outputs:
            url = self._sanitize(candidate)
            if url:
                if url not in self._reference_history:
                    self._reference_history.appendleft(url)
                break

    @staticmethod
    def _sanitize(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def seed_for_page(self, page_number: int) -> int | None:
        if self._base_seed is None:
            return None
        if self._config.locked_seed is not None:
            return self._base_seed
        if self._config.vary_seed_per_page:
            offset = max(page_number - 1, 0)
            seed = (self._base_seed + offset) % self._seed_mod
            if seed <= 0:
                seed = self._base_seed or 1
            return seed
        return self._base_seed

    @staticmethod
    def _format_identity_notes(profile: KidProfile) -> str | None:
        bullets = profile.context_bullets()
        lines = [line.strip() for line in bullets if line and line.strip()]
        if not lines:
            return None
        return "\n".join(lines)

    @staticmethod
    def _summarize_supporting_details(details: str | None) -> list[str]:
        if not details:
            return []

        normalized: list[str] = []
        for raw in details.replace("\r", "\n").split("\n"):
            cleaned = raw.strip(" \t-•—")
            if cleaned:
                normalized.append(cleaned)

        if not normalized:
            normalized = [details.strip()]

        summaries: list[str] = []
        for line in normalized:
            if len(line) > 220:
                summaries.append(line[:217].rstrip() + "…")
            else:
                summaries.append(line)
        return summaries

    @staticmethod
    def _derive_consistent_seed(
        *,
        profile: KidProfile,
        reference_image: PathLike,
    ) -> int:
        hasher = hashlib.blake2b(digest_size=8)
        hasher.update(profile.name.strip().lower().encode("utf-8"))

        for value in (
            profile.nickname,
            profile.gender,
            profile.favorite_theme,
            profile.desired_takeaway,
            profile.personal_notes,
        ):
            if value:
                hasher.update(str(value).strip().lower().encode("utf-8"))

        if profile.age is not None:
            hasher.update(str(profile.age).encode("utf-8"))

        if profile.hobbies:
            joined = ",".join(sorted(item.strip().lower() for item in profile.hobbies))
            hasher.update(joined.encode("utf-8"))

        hasher.update(str(reference_image).strip().encode("utf-8"))

        seed = int.from_bytes(hasher.digest(), "big") % IllustrationContinuityState._SEED_MOD
        return seed or 1


