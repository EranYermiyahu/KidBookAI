"""
Orchestrates the full KidBookAI pipeline from profile to story and images.
"""

from __future__ import annotations

from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from src.ai_generation import ReplicateImageGenerator
from src.common import CompletionCallable
from src.story_generation import (
    KidProfile,
    SceneDescription,
    SceneDescriptionGenerator,
    StoryOutlineGenerator,
    StoryPage,
    StoryPageSplitter,
)

from .continuity import (
    GenerationContinuityDirectives,
    IllustrationContinuityConfig,
    IllustrationContinuityState,
)

ProgressCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class PageAsset:
    """Represents all data for a single story page."""

    page: StoryPage
    scene: SceneDescription
    image_outputs: Sequence[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page.page_number,
            "title": self.page.title,
            "story_text": self.page.story_text,
            "scene_description": self.scene.scene_description,
            "supporting_details": self.scene.supporting_details,
            "image_outputs": list(self.image_outputs),
        }


@dataclass
class StoryPackage:
    """Aggregated output of the KidBookAI pipeline."""

    profile: KidProfile
    story_markdown: str
    pages: list[PageAsset]

    def to_dict(self) -> dict[str, Any]:
        return {
            "child_profile": {
                "name": self.profile.name,
                "age": self.profile.age,
                "gender": self.profile.gender,
                "nickname": self.profile.nickname,
                "hobbies": list(self.profile.hobbies),
                "favorite_theme": self.profile.favorite_theme,
                "desired_takeaway": self.profile.desired_takeaway,
                "personal_notes": self.profile.personal_notes,
                "story_language": self.profile.story_language,
                "guardian_name": self.profile.guardian_name,
            },
            "story_markdown": self.story_markdown,
            "pages": [asset.to_dict() for asset in self.pages],
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StoryPackage":
        if "child_profile" not in payload:
            raise ValueError("Story package payload must include 'child_profile'.")
        if "pages" not in payload:
            raise ValueError("Story package payload must include 'pages'.")

        profile = KidProfile.from_mapping(payload["child_profile"])
        story_markdown = str(payload.get("story_markdown", "")).strip()

        pages_payload = payload.get("pages", [])
        pages: list[PageAsset] = []
        for entry in pages_payload:
            try:
                page_number = int(entry["page_number"])
                title = str(entry["title"]).strip()
                story_text = str(entry["story_text"]).strip()
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid page entry: {entry}") from exc

            story_page = StoryPage(page_number=page_number, title=title, story_text=story_text)
            scene = SceneDescription(
                page_number=page_number,
                scene_description=str(entry.get("scene_description", "")).strip(),
                supporting_details=str(entry.get("supporting_details", "")).strip(),
            )
            raw_outputs = entry.get("image_outputs", [])
            image_outputs = tuple(normalize_image_outputs(raw_outputs))
            pages.append(PageAsset(page=story_page, scene=scene, image_outputs=image_outputs))

        return cls(profile=profile, story_markdown=story_markdown, pages=pages)

    @classmethod
    def from_yaml(cls, source: str | Path) -> "StoryPackage":
        path = Path(source)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("Story package YAML must deserialize to a mapping.")
        return cls.from_dict(data)


class KidBookAIOrchestrator:
    """
    High-level coordinator that chains together story and image generation workflows.
    """

    def __init__(
        self,
        *,
        story_generator: StoryOutlineGenerator | None = None,
        page_splitter: StoryPageSplitter | None = None,
        scene_generator: SceneDescriptionGenerator | None = None,
        image_generator: ReplicateImageGenerator | None = None,
        story_model: str | None = None,
        page_model: str | None = None,
        scene_model: str | None = None,
        story_api_key: str | None = None,
        page_api_key: str | None = None,
        scene_api_key: str | None = None,
        completion_fn: CompletionCallable | None = None,
        continuity_config: IllustrationContinuityConfig | None = None,
    ) -> None:
        self._story_generator = story_generator or StoryOutlineGenerator(
            api_key=story_api_key,
            model=story_model,
            completion_fn=completion_fn,
        )
        self._page_splitter = page_splitter or StoryPageSplitter(
            api_key=page_api_key,
            model=page_model,
            completion_fn=completion_fn,
        )
        self._scene_generator = scene_generator or SceneDescriptionGenerator(
            api_key=scene_api_key,
            model=scene_model,
            completion_fn=completion_fn,
        )
        self._image_generator = image_generator or ReplicateImageGenerator()
        self._continuity_config = continuity_config

    def run_from_profile_mapping(
        self,
        profile_data: Mapping[str, Any],
        *,
        desired_pages: int | None = None,
        reference_image: str | Path,
        image_kwargs: Mapping[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> StoryPackage:
        """
        Complete pipeline from raw profile mapping to story, pages, scenes, and images.
        """
        self._notify(progress_callback, "profile:parsing", source="mapping")
        profile = KidProfile.from_mapping(profile_data)
        self._notify(
            progress_callback,
            "profile:ready",
            name=profile.name,
            story_language=profile.story_language,
        )
        return self._run_pipeline(
            profile=profile,
            desired_pages=desired_pages,
            reference_image=reference_image,
            image_kwargs=image_kwargs,
            progress_callback=progress_callback,
        )

    def run_from_profile_file(
        self,
        profile_path: Path | str,
        *,
        desired_pages: int | None = None,
        reference_image: str | Path,
        image_kwargs: Mapping[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> StoryPackage:
        """
        Load profile data from a YAML or JSON file and run the pipeline.
        """
        profile_path = Path(profile_path)
        self._notify(progress_callback, "profile:parsing", source=str(profile_path))
        data = _load_mapping_file(profile_path)
        return self.run_from_profile_mapping(
            data,
            desired_pages=desired_pages,
            reference_image=reference_image,
            image_kwargs=image_kwargs,
            progress_callback=progress_callback,
        )

    def _run_pipeline(
        self,
        *,
        profile: KidProfile,
        desired_pages: int | None,
        reference_image: str | Path,
        image_kwargs: Mapping[str, Any] | None,
        progress_callback: ProgressCallback | None,
    ) -> StoryPackage:
        self._notify(progress_callback, "story:generating")
        story_markdown = self._story_generator.generate_story(profile)
        self._notify(
            progress_callback,
            "story:generated",
            word_count=len(story_markdown.split()),
        )

        self._notify(progress_callback, "pages:splitting")
        pages = self._page_splitter.split_story(
            story_markdown,
            profile=profile,
            desired_pages=desired_pages,
        )
        total_pages = len(pages)
        self._notify(progress_callback, "pages:ready", total_pages=total_pages)

        page_assets = self._generate_page_assets(
            profile=profile,
            pages=pages,
            reference_image=reference_image,
            image_kwargs=image_kwargs or {},
            progress_callback=progress_callback,
        )

        self._notify(
            progress_callback,
            "pipeline:packaging",
            total_pages=len(page_assets),
            profile_name=profile.name,
        )

        package = StoryPackage(
            profile=profile,
            story_markdown=story_markdown,
            pages=page_assets,
        )

        self._notify(
            progress_callback,
            "pipeline:complete",
            total_pages=len(page_assets),
            profile_name=profile.name,
        )

        return package

    def _generate_page_assets(
        self,
        *,
        profile: KidProfile,
        pages: Iterable[StoryPage],
        reference_image: str | Path,
        image_kwargs: Mapping[str, Any],
        progress_callback: ProgressCallback | None,
    ) -> list[PageAsset]:
        assets: list[PageAsset] = []
        pages_list = list(pages)
        total_pages = len(pages_list)
        continuity_state = IllustrationContinuityState(
            profile=profile,
            base_reference_image=reference_image,
            config=self._continuity_config,
        )
        for index, page in enumerate(pages_list, start=1):
            self._notify(
                progress_callback,
                "page:processing",
                page_number=page.page_number,
                page_index=index,
                total_pages=total_pages,
                title=page.title,
            )
            scene = self._scene_generator.page_to_scene(page, profile=profile)
            directives = continuity_state.build_directives(
                scene=scene,
                page_number=page.page_number,
            )

            page_image_kwargs: dict[str, Any] = dict(image_kwargs)
            for key, value in directives.model_overrides.items():
                page_image_kwargs.setdefault(key, value)
            if (
                directives.reference_history_parameter
                and directives.additional_reference_images
                and directives.reference_history_parameter not in page_image_kwargs
            ):
                page_image_kwargs[
                    directives.reference_history_parameter
                ] = list(directives.additional_reference_images)

            image_outputs = self._invoke_image_generation(
                profile=profile,
                scene=scene,
                reference_image=directives.primary_reference_image,
                continuity_directives=directives,
                image_kwargs=page_image_kwargs,
            )
            assets.append(PageAsset(page=page, scene=scene, image_outputs=image_outputs))
            continuity_state.record_generation(scene=scene, image_outputs=image_outputs)
            self._notify(
                progress_callback,
                "page:done",
                page_number=page.page_number,
                page_index=index,
                total_pages=total_pages,
            )
        return assets

    def _invoke_image_generation(
        self,
        *,
        profile: KidProfile,
        scene: SceneDescription,
        reference_image: str | Path,
        image_kwargs: Mapping[str, Any],
        continuity_directives: GenerationContinuityDirectives,
    ) -> Sequence[str]:
        model_kwargs = dict(image_kwargs)
        camera_shot = model_kwargs.pop("camera_shot", None)

        outputs = self._image_generator.generate_image(
            kid_name=profile.name,
            scene_description=scene.scene_description,
            input_image=reference_image,
            camera_shot=camera_shot,
            identity_traits=continuity_directives.identity_notes,
            continuity_notes=continuity_directives.continuity_notes,
            supporting_cast_notes=continuity_directives.supporting_cast_notes,
            **model_kwargs,
        )
        return normalize_image_outputs(outputs)

    @staticmethod
    def _notify(
        callback: ProgressCallback | None,
        stage: str,
        **payload: Any,
    ) -> None:
        if callback is not None:
            callback(stage, payload)


def _load_mapping_file(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    if path.suffix.lower() == ".json":
        import json

        return json.loads(text)
    raise ValueError("Unsupported profile file format. Use YAML or JSON.")


def normalize_image_outputs(raw: Any) -> list[str]:
    """
    Normalize the image outputs returned by Replicate into a list of URL strings.
    """

    if raw is None:
        return []

    if isinstance(raw, str):
        return [raw]

    if isinstance(raw, bytes):
        return [raw.decode("utf-8", errors="ignore")]

    if isinstance(raw, IterableABC):
        collected = list(raw)
        if not collected:
            return []

        if all(isinstance(item, str) and len(item) == 1 for item in collected):
            return ["".join(collected)]

        normalized: list[str] = []
        for item in collected:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, bytes):
                normalized.append(item.decode("utf-8", errors="ignore"))
            elif isinstance(item, IterableABC):
                nested = normalize_image_outputs(item)
                normalized.extend(nested)
            elif item is not None:
                normalized.append(str(item))
        return normalized

    return [str(raw)]
