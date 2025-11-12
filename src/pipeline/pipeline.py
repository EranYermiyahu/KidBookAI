"""
Orchestrates the full KidBookAI pipeline from profile to story and images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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

    def run_from_profile_mapping(
        self,
        profile_data: Mapping[str, Any],
        *,
        desired_pages: int | None = None,
        reference_image: str | Path,
        image_kwargs: Mapping[str, Any] | None = None,
    ) -> StoryPackage:
        """
        Complete pipeline from raw profile mapping to story, pages, scenes, and images.
        """
        profile = KidProfile.from_mapping(profile_data)
        return self._run_pipeline(
            profile=profile,
            desired_pages=desired_pages,
            reference_image=reference_image,
            image_kwargs=image_kwargs,
        )

    def run_from_profile_file(
        self,
        profile_path: Path | str,
        *,
        desired_pages: int | None = None,
        reference_image: str | Path,
        image_kwargs: Mapping[str, Any] | None = None,
    ) -> StoryPackage:
        """
        Load profile data from a YAML or JSON file and run the pipeline.
        """
        profile_path = Path(profile_path)
        data = _load_mapping_file(profile_path)
        return self.run_from_profile_mapping(
            data,
            desired_pages=desired_pages,
            reference_image=reference_image,
            image_kwargs=image_kwargs,
        )

    def _run_pipeline(
        self,
        *,
        profile: KidProfile,
        desired_pages: int | None,
        reference_image: str | Path,
        image_kwargs: Mapping[str, Any] | None,
    ) -> StoryPackage:
        story_markdown = self._story_generator.generate_story(profile)
        pages = self._page_splitter.split_story(
            story_markdown,
            profile=profile,
            desired_pages=desired_pages,
        )
        page_assets = self._generate_page_assets(
            profile=profile,
            pages=pages,
            reference_image=reference_image,
            image_kwargs=image_kwargs or {},
        )

        return StoryPackage(
            profile=profile,
            story_markdown=story_markdown,
            pages=page_assets,
        )

    def _generate_page_assets(
        self,
        *,
        profile: KidProfile,
        pages: Iterable[StoryPage],
        reference_image: str | Path,
        image_kwargs: Mapping[str, Any],
    ) -> list[PageAsset]:
        assets: list[PageAsset] = []
        for page in pages:
            scene = self._scene_generator.page_to_scene(page, profile=profile)
            image_outputs = self._invoke_image_generation(
                profile=profile,
                scene=scene,
                reference_image=reference_image,
                image_kwargs=image_kwargs,
            )
            assets.append(PageAsset(page=page, scene=scene, image_outputs=image_outputs))
        return assets

    def _invoke_image_generation(
        self,
        *,
        profile: KidProfile,
        scene: SceneDescription,
        reference_image: str | Path,
        image_kwargs: Mapping[str, Any],
    ) -> Sequence[str]:
        outputs = self._image_generator.generate_image(
            kid_name=profile.name,
            scene_description=scene.scene_description,
            input_image=reference_image,
            **image_kwargs,
        )
        return list(outputs)


def _load_mapping_file(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    if path.suffix.lower() == ".json":
        import json

        return json.loads(text)
    raise ValueError("Unsupported profile file format. Use YAML or JSON.")

