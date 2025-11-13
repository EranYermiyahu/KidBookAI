"""
Render a KidBookAI story package YAML into a printable PDF.

Usage:
    python scripts/render_story_pdf.py \
        --package kidbook_package.yaml \
        --output kidbook_story.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

# Ensure project root is on the Python path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import StoryPackage, StorybookPDFBuilder  # noqa: E402
from src.pdf_generation.builder import PAGE_SIZES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a KidBookAI story package YAML into a storybook PDF."
    )
    parser.add_argument(
        "--package",
        required=True,
        help="Path to the story package YAML (output of run_full_pipeline.py).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination PDF file path.",
    )
    parser.add_argument(
        "--page-size",
        choices=sorted(PAGE_SIZES.keys()),
        default="square",
        help="Page size to render (default: square).",
    )
    parser.add_argument(
        "--margin-mm",
        type=float,
        default=18.0,
        help="Page margin in millimetres (default: 18).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for downloading illustration assets (default: 30).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    page_size: tuple[float, float] = PAGE_SIZES[args.page_size]
    package = StoryPackage.from_yaml(args.package)

    builder = StorybookPDFBuilder(
        page_size=page_size,
        margin_mm=args.margin_mm,
        request_timeout=args.timeout,
    )
    builder.build(package, args.output)

    print(f"Rendered storybook PDF to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


