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
from pathlib import Path
from typing import Any, Dict

import sys

from tqdm.auto import tqdm

# Ensure project root is on the Python path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import KidBookAIOrchestrator


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
    return parser.parse_args()


def parse_image_kwargs(pairs: list[str]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid --image-arg '{pair}', expected KEY=VALUE.")
        key, value = pair.split("=", 1)
        kwargs[key] = value
    return kwargs


def main() -> int:
    args = parse_args()

    image_kwargs = parse_image_kwargs(args.image_arg)
    orchestrator = KidBookAIOrchestrator()
    tracker = ProgressTracker()

    try:
        package = orchestrator.run_from_profile_file(
            Path(args.profile),
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


if __name__ == "__main__":
    raise SystemExit(main())

