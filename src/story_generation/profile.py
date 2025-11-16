"""
Structured representations of the personalized kid information gathered from the form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


def _normalize_hobbies(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",")]
    elif isinstance(value, Sequence):
        parts = [str(item).strip() for item in value]
    else:
        raise TypeError("hobbies must be a string or sequence of strings.")

    return tuple(filter(None, parts))


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None

    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected an integer-compatible value for age, got {value!r}") from exc


def _normalize_supporting_characters(value: Any) -> dict[str, str]:
    if value is None:
        return {}

    def _add_entry(
        target: dict[str, str],
        name: Any,
        description: Any,
    ) -> None:
        name_text = str(name).strip() if name is not None else ""
        desc_text = str(description).strip() if description is not None else ""
        if name_text and desc_text:
            target[name_text] = desc_text

    if isinstance(value, Mapping):
        result: dict[str, str] = {}
        for name, details in value.items():
            if isinstance(details, Mapping):
                description = details.get("description") or details.get("notes")
                _add_entry(result, name, description)
            else:
                _add_entry(result, name, details)
        return result

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result: dict[str, str] = {}
        for item in value:
            if isinstance(item, Mapping):
                name = item.get("name")
                description = item.get("description") or item.get("notes")
                _add_entry(result, name, description)
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                if len(item) >= 2:
                    _add_entry(result, item[0], item[1])
            elif isinstance(item, str):
                if ":" in item:
                    name, description = item.split(":", 1)
                    _add_entry(result, name, description)
        return result

    if isinstance(value, str):
        result: dict[str, str] = {}
        for line in value.replace("\r", "\n").split("\n"):
            if ":" in line:
                name, description = line.split(":", 1)
                _add_entry(result, name, description)
        return result

    raise TypeError("supporting_characters must be a mapping, sequence, or colon-delimited string.")


def _normalize_continuity_notes(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()

    def _normalize_string(text: str) -> list[str]:
        normalized: list[str] = []
        for raw in text.replace("\r", "\n").split("\n"):
            cleaned = raw.strip(" \t-•—")
            if cleaned:
                normalized.append(cleaned)
        return normalized

    if isinstance(value, str):
        return tuple(_normalize_string(value))

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        notes: list[str] = []
        for item in value:
            if isinstance(item, str):
                notes.extend(_normalize_string(item))
            elif isinstance(item, Mapping):
                description = item.get("description") or item.get("note") or item.get("value")
                if isinstance(description, str):
                    notes.extend(_normalize_string(description))
            elif item is not None:
                notes.extend(_normalize_string(str(item)))
        return tuple(notes)

    return tuple(_normalize_string(str(value)))


@dataclass(frozen=True)
class KidProfile:
    """
    Canonical representation of the personalized kid inputs.

    Attributes
    ----------
    name:
        Child's primary name (required).
    age:
        Age in years, if provided.
    gender:
        Gender or pronoun preference. Used to craft respectful phrasing.
    nickname:
        Familiar nickname to sprinkle into the narrative.
    hobbies:
        Tuple of hobbies/interests that can colour the story.
    favorite_theme:
        High-level adventure theme chosen by the parent/guardian.
    desired_takeaway:
        Moral, lesson, or emotional takeaway requested for the ending.
    personal_notes:
        Free-form notes with any additional context.
    story_language:
        Preferred language for narrative output (defaults to English).
    guardian_name:
        Optional parent/guardian name for acknowledgements.
    """

    name: str
    age: int | None = None
    gender: str | None = None
    nickname: str | None = None
    hobbies: tuple[str, ...] = ()
    favorite_theme: str | None = None
    desired_takeaway: str | None = None
    personal_notes: str | None = None
    story_language: str = "English"
    guardian_name: str | None = None
    identity_traits: str | None = None
    supporting_characters: Mapping[str, str] = field(default_factory=dict)
    continuity_notes: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "KidProfile":
        """
        Build a profile from a dict-like object (e.g., parsed JSON/YAML).
        """
        if "name" not in data or not str(data["name"]).strip():
            raise ValueError("Profile data must include a non-empty 'name' field.")

        hobbies = _normalize_hobbies(data.get("hobbies"))

        return cls(
            name=str(data["name"]).strip(),
            age=_coerce_optional_int(data.get("age")),
            gender=_coerce_optional_str(
                data.get("gender") or data.get("sex") or data.get("pronouns")
            ),
            nickname=_coerce_optional_str(data.get("nickname")),
            hobbies=hobbies,
            favorite_theme=_coerce_optional_str(
                data.get("preferred_theme") or data.get("favorite_theme")
            ),
            desired_takeaway=_coerce_optional_str(
                data.get("desired_takeaway") or data.get("lesson")
            ),
            personal_notes=_coerce_optional_str(data.get("personal_notes")),
            story_language=_coerce_optional_str(
                data.get("story_language") or data.get("language")
            )
            or "English",
            guardian_name=_coerce_optional_str(
                data.get("guardian_name")
                or data.get("parent_name")
                or data.get("caregiver_name")
            ),
            identity_traits=_coerce_optional_str(
                data.get("identity_traits")
                or data.get("visual_identity_notes")
                or data.get("identity_notes")
            ),
            supporting_characters=_normalize_supporting_characters(
                data.get("supporting_characters")
                or data.get("recurring_characters")
            ),
            continuity_notes=_normalize_continuity_notes(
                data.get("continuity_notes")
                or data.get("wardrobe_notes")
                or data.get("visual_continuity_notes")
            ),
        )

    def context_bullets(self) -> list[str]:
        """
        Produce bullet-friendly lines describing the child, for prompt conditioning.
        """
        bullets: list[str] = [f"Name: {self.name}"]

        if self.nickname:
            bullets.append(f"Nickname: {self.nickname}")

        if self.age is not None:
            bullets.append(f"Age: {self.age}")

        if self.gender:
            bullets.append(f"Gender/pronouns: {self.gender}")

        if self.hobbies:
            joined = ", ".join(self.hobbies)
            bullets.append(f"Hobbies/interests: {joined}")

        if self.favorite_theme:
            bullets.append(f"Preferred theme: {self.favorite_theme}")

        if self.desired_takeaway:
            bullets.append(f"Desired takeaway: {self.desired_takeaway}")

        if self.personal_notes:
            bullets.append(f"Special notes: {self.personal_notes}")

        if self.guardian_name:
            bullets.append(f"Guardian name: {self.guardian_name}")

        if self.identity_traits:
            bullets.append(f"Identity traits: {self.identity_traits}")

        if self.continuity_notes:
            for note in self.continuity_notes:
                bullets.append(f"Continuity note: {note}")

        bullets.append(f"Story language: {self.story_language}")

        return bullets

    def summary_for_prompt(self) -> str:
        """
        Format the profile as a readable block suitable for LLM prompting.
        """
        lines = self.context_bullets()
        return "\n".join(f"- {line}" for line in lines)

