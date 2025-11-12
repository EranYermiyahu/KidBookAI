"""
Utility script to exercise the Replicate integration with a sample scene.

Usage:
    python scripts/run_storybook_example.py \
        --kid-name "Laura" \
        --scene "Laura rides a gentle dragon through a sunset sky sprinkled with stars."

Environment variables:
    REPLICATE_API_TOKEN  - required unless you pass --api-token
    REPLICATE_MODEL      - required unless you pass --model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from src.ai_generation import ReplicateImageGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "example_images" / "laura_girl.jpg"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a personalized KidBookAI illustration via Replicate."
    )
    parser.add_argument(
        "--kid-name",
        required=True,
        help="Name of the child featured in the storybook illustration.",
    )
    parser.add_argument(
        "--scene",
        required=True,
        help="Scene description narrating what should be illustrated.",
    )
    parser.add_argument(
        "--image-path",
        default=str(DEFAULT_IMAGE_PATH),
        help=f"Reference image path or URL. Defaults to {DEFAULT_IMAGE_PATH}",
    )
    parser.add_argument(
        "--api-token",
        default=None,
        help="Optional Replicate API token override (otherwise uses environment variable).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional Replicate model identifier override (owner/model:version).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    generator = ReplicateImageGenerator(
        api_token=args.api_token,
        model_identifier=args.model,
    )

    print("Running generation with the following parameters:")
    print(f"  Kid name: {args.kid_name}")
    print(f"  Scene   : {args.scene}")
    print(f"  Image   : {args.image_path}")
    print(f"  Model   : {generator.model_identifier}")

    results: Iterable[str] = generator.generate_image(
        kid_name=args.kid_name,
        scene_description=args.scene,
        input_image=args.image_path,
    )

    print("\nReplicate output:")
    for item in results:
        print(f"  {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

