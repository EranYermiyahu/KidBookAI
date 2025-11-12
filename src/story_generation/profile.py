"""
Structured representations of the personalized kid information gathered from the form.
"""

from __future__ import annotations

from dataclasses import dataclass
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

        bullets.append(f"Story language: {self.story_language}")

        return bullets

    def summary_for_prompt(self) -> str:
        """
        Format the profile as a readable block suitable for LLM prompting.
        """
        lines = self.context_bullets()
        return "\n".join(f"- {line}" for line in lines)

