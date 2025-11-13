"""
High-level utilities for rendering KidBookAI story packages into printable PDFs.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional, Sequence

import requests
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Paragraph

from src.pipeline.pipeline import (
    PageAsset,
    StoryPackage,
    normalize_image_outputs,
)


@dataclass(frozen=True)
class PageLayoutConfig:
    text_background: colors.Color
    image_background: colors.Color
    cover_background: colors.Color
    accent_color: colors.Color
    text_color: colors.Color
    caption_color: colors.Color


DEFAULT_LAYOUT = PageLayoutConfig(
    text_background=colors.HexColor("#F5F1FF"),
    image_background=colors.HexColor("#E8F5FF"),
    cover_background=colors.HexColor("#6C4FD3"),
    accent_color=colors.HexColor("#FFB347"),
    text_color=colors.HexColor("#2F2A40"),
    caption_color=colors.HexColor("#4B506D"),
)


PAGE_SIZES = {
    "a4": A4,
    "letter": LETTER,
    "square": (8 * inch, 8 * inch),
}


class StorybookPDFBuilder:
    """
    Render KidBookAI story packages into production-ready printable PDFs.

    The builder creates:
      * A cover page highlighting the story title, vibe, and summary.
      * Alternating spreads where odd-numbered pages carry text and even pages showcase
        the corresponding illustration.
    """

    def __init__(
        self,
        *,
        page_size: tuple[float, float] = PAGE_SIZES["square"],
        margin_mm: float = 18.0,
        layout: PageLayoutConfig = DEFAULT_LAYOUT,
        request_timeout: float = 30.0,
    ) -> None:
        self.page_size = page_size
        self.margin = margin_mm * mm
        self.layout = layout
        self.request_timeout = request_timeout

        self.body_font, self.body_bold_font = self._configure_story_fonts()

        self.title_style = ParagraphStyle(
            name="StoryTitle",
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=32,
            alignment=TA_CENTER,
            textColor=colors.white,
            spaceAfter=12,
        )
        self.subtitle_style = ParagraphStyle(
            name="StorySubtitle",
            fontName="Helvetica",
            fontSize=16,
            leading=20,
            alignment=TA_CENTER,
            textColor=colors.white,
            spaceAfter=18,
        )
        self.body_title_style = ParagraphStyle(
            name="BodyTitle",
            fontName=self.body_bold_font,
            fontSize=28,
            leading=32,
            alignment=TA_CENTER,
            textColor=self.layout.text_color,
            spaceAfter=14,
        )
        self.body_style = ParagraphStyle(
            name="Body",
            fontName=self.body_font,
            fontSize=18,
            leading=27,
            alignment=TA_JUSTIFY,
            textColor=self.layout.text_color,
            spaceAfter=16,
        )
        self.caption_style = ParagraphStyle(
            name="ImageCaption",
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            alignment=TA_CENTER,
            textColor=self.layout.caption_color,
        )
        self.footer_style = ParagraphStyle(
            name="Footer",
            fontName="Helvetica-Oblique",
            fontSize=10,
            leading=12,
            alignment=TA_CENTER,
            textColor=self.layout.caption_color,
        )

    def build_from_yaml(self, package_path: Path | str, output_path: Path | str) -> None:
        package = StoryPackage.from_yaml(package_path)
        self.build(package, output_path)

    def build(self, package: StoryPackage, output_path: Path | str) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        pdf = canvas.Canvas(str(output_file), pagesize=self.page_size)
        width, height = self.page_size

        self._draw_cover_page(pdf, package, width, height)

        for page_asset in package.pages:
            self._draw_text_page(pdf, package, page_asset, width, height)
            self._draw_image_page(pdf, page_asset, width, height)

        pdf.save()

    # ------------------------------------------------------------------ cover rendering

    def _draw_cover_page(
        self,
        pdf: canvas.Canvas,
        package: StoryPackage,
        width: float,
        height: float,
    ) -> None:
        pdf.setFillColor(self.layout.cover_background)
        pdf.rect(0, 0, width, height, stroke=0, fill=1)

        title = self._extract_primary_title(package.story_markdown) or package.profile.name
        vibe = self._extract_section(package.story_markdown, "## Vibe Check")
        summary = self._extract_section(package.story_markdown, "## Summary")

        frame = Frame(
            self.margin,
            self.margin,
            width - 2 * self.margin,
            height - 2 * self.margin,
            showBoundary=0,
        )

        intro = [
            Paragraph(title, self.title_style),
            Paragraph(f"A personalised adventure starring {package.profile.name}", self.subtitle_style),
        ]
        if vibe:
            intro.append(Paragraph(vibe.replace("\n", "<br/>"), self.subtitle_style))
        if summary:
            intro.append(
                Paragraph(
                    summary.replace("\n", "<br/>"),
                    ParagraphStyle(
                        "Summary",
                        parent=self.subtitle_style,
                        fontSize=14,
                        leading=18,
                    ),
                )
            )

        frame.addFromList(intro, pdf)
        pdf.showPage()

    # ------------------------------------------------------------------ text pages

    def _draw_text_page(
        self,
        pdf: canvas.Canvas,
        package: StoryPackage,
        page_asset: PageAsset,
        width: float,
        height: float,
    ) -> None:
        pdf.setFillColor(self.layout.text_background)
        pdf.rect(0, 0, width, height, stroke=0, fill=1)

        bubble_width = width - (self.margin * 2 * 0.6)
        bubble_height = height - (self.margin * 2 * 0.6)
        bubble_x = (width - bubble_width) / 2
        bubble_y = (height - bubble_height) / 2

        pdf.saveState()
        pdf.setFillColor(self._lighten(self.layout.accent_color, 0.75))
        pdf.roundRect(bubble_x, bubble_y, bubble_width, bubble_height, 26, stroke=0, fill=1)
        pdf.restoreState()

        self._draw_sparkles(pdf, bubble_x, bubble_y, bubble_width, bubble_height)

        content_width = bubble_width - (self.margin * 2 * 0.3)
        content_height = bubble_height - (self.margin * 2 * 0.3)
        frame_x = bubble_x + (bubble_width - content_width) / 2
        frame_y = bubble_y + (bubble_height - content_height) / 2

        frame = Frame(
            frame_x,
            frame_y,
            content_width,
            content_height,
            showBoundary=0,
        )

        story_paragraphs = [Paragraph(page_asset.page.title, self.body_title_style)]

        for block in filter(None, (paragraph.strip() for paragraph in page_asset.page.story_text.split("\n\n"))):
            story_paragraphs.append(
                Paragraph(block.replace("\n", "<br/>"), self.body_style)
            )

        frame.addFromList(story_paragraphs, pdf)

        footer_text = f"Page {page_asset.page.page_number} • {package.profile.name}'s Story"
        self._draw_footer(pdf, footer_text, width)
        pdf.showPage()

    # ------------------------------------------------------------------ image pages

    def _draw_image_page(
        self,
        pdf: canvas.Canvas,
        page_asset: PageAsset,
        width: float,
        height: float,
    ) -> None:
        pdf.setFillColor(self.layout.image_background)
        pdf.rect(0, 0, width, height, stroke=0, fill=1)

        urls = normalize_image_outputs(page_asset.image_outputs)
        image_reader = self._fetch_image(urls[0]) if urls else None

        if image_reader is not None:
            img_width, img_height = image_reader.getSize()
            scale = max(width / img_width, height / img_height)
            draw_width = img_width * scale
            draw_height = img_height * scale
            x = (width - draw_width) / 2
            y = (height - draw_height) / 2
            pdf.drawImage(
                image_reader,
                x,
                y,
                draw_width,
                draw_height,
                preserveAspectRatio=True,
                mask="auto",
            )

        footer_text = f"Illustration for Page {page_asset.page.page_number}"
        self._draw_footer(pdf, footer_text, width)
        pdf.showPage()

    # ------------------------------------------------------------------ helpers

    def _draw_footer(self, pdf: canvas.Canvas, text: str, width: float) -> None:
        footer_frame = Frame(
            self.margin,
            10,
            width - 2 * self.margin,
            20,
            showBoundary=0,
        )
        footer_frame.addFromList([Paragraph(text, self.footer_style)], pdf)

    def _fetch_image(self, url: str) -> Optional[ImageReader]:
        try:
            response = requests.get(url, timeout=self.request_timeout)
            response.raise_for_status()
        except requests.RequestException:
            return None
        return ImageReader(BytesIO(response.content))

    @staticmethod
    def _extract_primary_title(markdown: str) -> str | None:
        for line in markdown.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped.lstrip("# ").strip()
        return None

    @staticmethod
    def _extract_section(markdown: str, heading: str) -> str:
        lines = markdown.splitlines()
        capture = False
        collected: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                if capture:
                    break
                if stripped.lower() == heading.lower():
                    capture = True
                    continue
            if capture:
                collected.append(line)
        return "\n".join(collected).strip()

    def _draw_sparkles(
        self,
        pdf: canvas.Canvas,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> None:
        sparkles = [
            (x + width * 0.18, y + height * 0.82, 6),
            (x + width * 0.82, y + height * 0.78, 9),
            (x + width * 0.25, y + height * 0.32, 5),
            (x + width * 0.74, y + height * 0.28, 6),
            (x + width * 0.5, y + height * 0.9, 4),
        ]
        pdf.saveState()
        pdf.setFillColor(self._lighten(self.layout.accent_color, 0.6))
        for cx, cy, radius in sparkles:
            pdf.circle(cx, cy, radius, stroke=0, fill=1)
        pdf.restoreState()

    @staticmethod
    def _lighten(color: colors.Color, amount: float = 0.5) -> colors.Color:
        amount = max(0.0, min(amount, 1.0))
        r = color.red + (1 - color.red) * amount
        g = color.green + (1 - color.green) * amount
        b = color.blue + (1 - color.blue) * amount
        return colors.Color(r, g, b)

    def _configure_story_fonts(self) -> tuple[str, str]:
        playful_options = [
            (
                "ComicSansMS",
                "ComicSansMS-Bold",
                ["Comic Sans MS.ttf", "ComicSansMS.ttf"],
                ["Comic Sans MS Bold.ttf", "ComicSansMS-Bold.ttf"],
            ),
            (
                "ChalkboardSE-Light",
                "ChalkboardSE-Bold",
                ["ChalkboardSE-Light.ttf", "ChalkboardSE.ttc"],
                ["ChalkboardSE-Bold.ttf"],
            ),
        ]

        search_roots = [
            Path("/Library/Fonts"),
            Path("/System/Library/Fonts"),
            Path.home() / "Library" / "Fonts",
            Path("C:/Windows/Fonts"),
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
        ]

        for regular_name, bold_name, regular_candidates, bold_candidates in playful_options:
            regular_ready = self._register_font_if_available(regular_name, regular_candidates, search_roots)
            bold_ready = self._register_font_if_available(bold_name, bold_candidates, search_roots)
            if regular_ready and bold_ready:
                return regular_name, bold_name

        return "Helvetica", "Helvetica-Bold"

    @staticmethod
    def _register_font_if_available(
        font_name: str,
        candidate_filenames: Sequence[str],
        search_roots: Sequence[Path],
    ) -> bool:
        if font_name in pdfmetrics.getRegisteredFontNames():
            return True

        for root in search_roots:
            for candidate in candidate_filenames:
                font_path = root / candidate
                if font_path.exists():
                    try:
                        pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
                        return True
                    except Exception:
                        continue
        return False



