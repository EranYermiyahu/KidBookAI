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
from typing import Any

import sys

# Ensure project root is on the Python path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import KidBookAIOrchestrator


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

    package = orchestrator.run_from_profile_file(
        Path(args.profile),
        desired_pages=args.pages,
        reference_image=args.reference_image,
        image_kwargs=image_kwargs,
    )

    output_path = Path(args.output)
    output_path.write_text(package.to_yaml(), encoding="utf-8")
    print(f"Saved story package to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

