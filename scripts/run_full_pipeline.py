"""
CLI example to run the complete KidBookAI pipeline end-to-end.

Usage:
    python scripts/run_full_pipeline.py \
        --profile kid_profile.yaml \
        --reference-image example_images/laura_girl.jpg \
        --output kidbook_package.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from tqdm.auto import tqdm

# Ensure project root is on the Python path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (
    KidBookAIOrchestrator,
    build_illustration_continuity_config,
)
from src.story_generation import KidProfile


class ProgressTracker:
    """
    Provides user-friendly command-line progress updates for the KidBookAI pipeline.
    """

    def __init__(self) -> None:
        self._page_bar: tqdm | None = None

    def __call__(self, stage: str, payload: Dict[str, Any]) -> None:
        match stage:
            case "profile:parsing":
                source = payload.get("source")
                self._write(
                    f"[1/6] Loading kid profile data"
                    + (f" from {source!s}..." if source else "...")
                )
            case "profile:ready":
                name = payload.get("name", "the child")
                language = payload.get("story_language", "English")
                self._write(f"[1/6] Profile ready for {name} (primary language: {language}).")
            case "story:generating":
                self._write("[2/6] Generating the personalized story...")
            case "story:generated":
                word_count = payload.get("word_count")
                summary = (
                    f" (~{word_count} words)." if isinstance(word_count, int) and word_count > 0 else "."
                )
                self._write(f"[2/6] Story drafting complete{summary}")
            case "pages:splitting":
                self._write("[3/6] Splitting the story into illustrated beats...")
            case "pages:ready":
                total = payload.get("total_pages", 0)
                self._write(f"[4/6] Story divided into {total} pages. Generating scenes & images...")
                self._page_bar = tqdm(total=total, desc="Illustrated pages", unit="page")
            case "page:processing":
                if self._page_bar is not None:
                    page_number = payload.get("page_number")
                    title = payload.get("title") or ""
                    truncated = (title[:45] + "…") if len(title or "") > 45 else title
                    if page_number is not None:
                        self._page_bar.set_description(f"Page {page_number}: {truncated}")
            case "page:done":
                if self._page_bar is not None:
                    self._page_bar.update(1)
            case "pipeline:packaging":
                self._write("[5/6] Packaging story bundle...")
            case "pipeline:complete":
                self._write("[6/6] Pipeline complete.")
                self.close()

    def close(self) -> None:
        if self._page_bar is not None:
            self._page_bar.close()
            self._page_bar = None

    @staticmethod
    def _write(message: str) -> None:
        tqdm.write(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full KidBookAI generation pipeline.")
    parser.add_argument(
        "--profile",
        required=True,
        help="Path to the kid profile YAML/JSON file.",
    )
    parser.add_argument(
        "--reference-image",
        required=True,
        help="Path or URL to the child's reference image.",
    )
    parser.add_argument(
        "--output",
        default="kidbook_package.yaml",
        help="Output YAML file to store the story, scenes, and image outputs.",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=None,
        help="Optional override for desired page count (must fall within 12-18).",
    )
    parser.add_argument(
        "--image-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional key=value overrides forwarded to the Replicate model.",
    )
    parser.add_argument(
        "--auto-identity-notes",
        dest="automatic_identity_notes",
        action="store_true",
        default=True,
        help="Automatically extract identity traits from the reference image (default: enabled).",
    )
    parser.add_argument(
        "--no-auto-identity-notes",
        dest="automatic_identity_notes",
        action="store_false",
        help="Disable automatic identity extraction.",
    )
    parser.add_argument(
        "--identity-model",
        default=None,
        help="Override the multimodal model used for identity extraction.",
    )
    parser.add_argument(
        "--identity-api-key",
        default=None,
        help="Override the API key used for identity extraction.",
    )
    parser.add_argument(
        "--identity-note",
        action="append",
        default=[],
        help="Additional identity bullet to include in continuity prompts (repeatable).",
    )
    parser.add_argument(
        "--supporting-character",
        action="append",
        default=[],
        metavar="NAME=DESCRIPTION",
        help="Add or override a supporting character descriptor (repeatable).",
    )
    parser.add_argument(
        "--continuity-note",
        action="append",
        default=[],
        help="Extra static continuity note to keep consistent across pages (repeatable).",
    )
    parser.add_argument(
        "--reference-history-size",
        type=int,
        default=None,
        help="Override the number of previous illustrations reused as references.",
    )
    parser.add_argument(
        "--reference-history-param",
        default=None,
        help="Override the model input parameter name for reference image history.",
    )
    parser.add_argument(
        "--promote-latest-reference",
        dest="promote_latest_reference",
        action="store_true",
        default=None,
        help="Promote the freshest illustration as the next primary reference.",
    )
    parser.add_argument(
        "--no-promote-latest-reference",
        dest="promote_latest_reference",
        action="store_false",
        help="Do not promote the latest illustration; keep the original portrait primary.",
    )
    return parser.parse_args()


def parse_image_kwargs(pairs: list[str]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid --image-arg '{pair}', expected KEY=VALUE.")
        key, value = pair.split("=", 1)
        kwargs[key] = value
    return kwargs


def parse_supporting_characters(pairs: list[str]) -> dict[str, str]:
    characters: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid --supporting-character '{pair}', expected NAME=DESCRIPTION.")
        name, description = pair.split("=", 1)
        name = name.strip()
        description = description.strip()
        if not name or not description:
            raise ValueError(
                f"Supporting character entries must include both name and description: '{pair}'."
            )
        characters[name] = description
    return characters


def load_profile_mapping(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError("Unsupported profile file format. Use YAML or JSON.")

    if not isinstance(data, Dict):
        raise ValueError("Profile file must deserialize to a mapping.")
    return data


def main() -> int:
    args = parse_args()

    image_kwargs = parse_image_kwargs(args.image_arg)
    profile_path = Path(args.profile)
    profile_mapping = load_profile_mapping(profile_path)
    profile = KidProfile.from_mapping(profile_mapping)

    supporting_overrides = parse_supporting_characters(args.supporting_character)

    continuity_assembly = build_illustration_continuity_config(
        profile=profile,
        reference_image=args.reference_image,
        automatic_identity_notes=args.automatic_identity_notes,
        identity_note_overrides=args.identity_note,
        supporting_character_overrides=supporting_overrides,
        static_continuity_overrides=args.continuity_note,
        reference_history_size=args.reference_history_size,
        reference_history_parameter=args.reference_history_param,
        promote_latest_reference=args.promote_latest_reference,
        identity_model=args.identity_model,
        identity_api_key=args.identity_api_key,
    )

    orchestrator = KidBookAIOrchestrator(continuity_config=continuity_assembly.config)
    tracker = ProgressTracker()

    _print_continuity_summary(continuity_assembly)

    try:
        package = orchestrator.run_from_profile_mapping(
            profile_mapping,
            desired_pages=args.pages,
            reference_image=args.reference_image,
            image_kwargs=image_kwargs,
            progress_callback=tracker,
        )
    finally:
        tracker.close()

    output_path = Path(args.output)
    output_path.write_text(package.to_yaml(), encoding="utf-8")
    print(f"Saved story package to {output_path}")
    return 0


def _print_continuity_summary(assembly) -> None:
    tqdm.write("Continuity configuration summary:")
    if assembly.identity_notes:
        tqdm.write("  Identity cues:")
        for line in assembly.identity_notes.splitlines():
            tqdm.write(f"    {line}")
    else:
        tqdm.write("  Identity cues: (none)")

    if assembly.supporting_character_notes:
        tqdm.write("  Supporting characters:")
        for name, details in assembly.supporting_character_notes.items():
            tqdm.write(f"    - {name}: {details}")
    else:
        tqdm.write("  Supporting characters: (none)")

    if assembly.static_continuity_notes:
        tqdm.write("  Static continuity notes:")
        for note in assembly.static_continuity_notes:
            tqdm.write(f"    - {note}")
    else:
        tqdm.write("  Static continuity notes: (none)")

    if assembly.auto_identity_notes:
        tqdm.write("  (Auto identity extraction succeeded.)")


if __name__ == "__main__":
    raise SystemExit(main())

