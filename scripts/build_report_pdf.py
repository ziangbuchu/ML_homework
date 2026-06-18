"""Build the coursework PDF report from the Markdown source.

The script intentionally supports only the Markdown subset used by
reports/ML_homework_report.md so the final PDF is reproducible without a
LaTeX or pandoc installation.
"""

from __future__ import annotations

import argparse
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from PIL import Image


PAGE_WIDTH = 8.27
PAGE_HEIGHT = 11.69
LEFT_MARGIN = 0.68
RIGHT_MARGIN = 0.68
TOP_MARGIN = 0.62
BOTTOM_MARGIN = 0.62
BODY_SIZE = 10.5
SMALL_SIZE = 9.0
CODE_SIZE = 8.8


@dataclass(frozen=True)
class ImageRef:
    alt: str
    path: Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/ML_homework_report.md")
    parser.add_argument("--output", default="reports/ML_homework_report.pdf")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = PdfRenderer(output_path)
    render_markdown(input_path, renderer)
    renderer.close()
    print(output_path)


class PdfRenderer:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.font = FontProperties(fname=str(find_cjk_font()))
        self.mono_font = self.font
        self.pdf = PdfPages(output_path)
        self.page_number = 0
        self.fig: plt.Figure | None = None
        self.y = 0.0
        self.new_page()

    def close(self) -> None:
        self.save_page()
        self.pdf.close()

    def new_page(self) -> None:
        self.save_page()
        self.page_number += 1
        self.fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
        self.y = PAGE_HEIGHT - TOP_MARGIN

    def save_page(self) -> None:
        if self.fig is None:
            return
        self.fig.text(
            0.5,
            0.035,
            f"{self.page_number}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontproperties=self.font,
            color="#666666",
        )
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.fig = None

    def ensure_space(self, height: float) -> None:
        if self.y - height < BOTTOM_MARGIN:
            self.new_page()

    def add_space(self, height: float = 0.08) -> None:
        self.y -= height

    def add_heading(self, text: str, level: int) -> None:
        if level == 1:
            size = 19
            spacing = 0.34
            chars = 24
        elif level == 2:
            size = 15
            spacing = 0.24
            chars = 34
        else:
            size = 12.5
            spacing = 0.18
            chars = 44
        self.ensure_space(0.55)
        self.add_space(0.05 if level == 1 else 0.12)
        for line in wrap_text(clean_inline(text), chars):
            self.add_line(line, size=size, weight="bold")
        self.add_space(spacing)

    def add_paragraph(self, text: str, *, size: float = BODY_SIZE, indent: float = 0.0) -> None:
        chars = max(24, int(52 - indent * 7 - (size - BODY_SIZE) * 2))
        for line in wrap_text(clean_inline(text), chars):
            self.add_line(line, size=size, indent=indent)
        self.add_space(0.08)

    def add_bullet(self, text: str, *, checked: bool | None = None) -> None:
        marker = "•" if checked is None else ("☑" if checked else "☐")
        cleaned = clean_inline(text)
        lines = wrap_text(cleaned, 50)
        if not lines:
            return
        self.add_line(f"{marker} {lines[0]}", size=BODY_SIZE, indent=0.05)
        for line in lines[1:]:
            self.add_line(line, size=BODY_SIZE, indent=0.34)
        self.add_space(0.04)

    def add_code_block(self, lines: list[str]) -> None:
        line_height = 0.18
        height = max(0.34, line_height * len(lines) + 0.18)
        self.ensure_space(height)
        assert self.fig is not None
        box_left = LEFT_MARGIN
        box_bottom = self.y - height
        box_width = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
        ax = self.fig.add_axes(
            [
                box_left / PAGE_WIDTH,
                box_bottom / PAGE_HEIGHT,
                box_width / PAGE_WIDTH,
                height / PAGE_HEIGHT,
            ]
        )
        ax.set_facecolor("#f6f6f6")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#d0d0d0")
        y = 0.91
        for line in lines:
            ax.text(
                0.03,
                y,
                line,
                ha="left",
                va="top",
                fontsize=CODE_SIZE,
                fontproperties=self.mono_font,
                color="#222222",
            )
            y -= 0.16
        self.y -= height + 0.15

    def add_table(self, raw_lines: list[str]) -> None:
        rows = parse_table(raw_lines)
        if not rows:
            return
        row_count = len(rows)
        height = min(3.8, 0.34 * row_count + 0.25)
        self.ensure_space(height + 0.2)
        assert self.fig is not None
        ax = self.fig.add_axes(
            [
                LEFT_MARGIN / PAGE_WIDTH,
                (self.y - height) / PAGE_HEIGHT,
                (PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN) / PAGE_WIDTH,
                height / PAGE_HEIGHT,
            ]
        )
        ax.axis("off")
        table = ax.table(
            cellText=rows[1:],
            colLabels=rows[0],
            cellLoc="center",
            colLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.4)
        table.scale(1.0, 1.25)
        for (row, _col), cell in table.get_celld().items():
            cell.get_text().set_fontproperties(self.font)
            if row == 0:
                cell.set_facecolor("#efefef")
                cell.get_text().set_weight("bold")
            cell.set_edgecolor("#777777")
        self.y -= height + 0.18

    def add_image(self, image: ImageRef) -> None:
        if not image.path.exists():
            self.add_paragraph(f"[missing image: {image.path}]", size=SMALL_SIZE)
            return
        with Image.open(image.path) as handle:
            width, height = handle.size
        max_width = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN
        display_width = max_width
        display_height = display_width * height / width
        if display_height > 3.25:
            display_height = 3.25
            display_width = display_height * width / height
        total_height = display_height + 0.36
        self.ensure_space(total_height)
        assert self.fig is not None
        left = LEFT_MARGIN + (max_width - display_width) / 2
        bottom = self.y - display_height
        ax = self.fig.add_axes(
            [
                left / PAGE_WIDTH,
                bottom / PAGE_HEIGHT,
                display_width / PAGE_WIDTH,
                display_height / PAGE_HEIGHT,
            ]
        )
        with Image.open(image.path) as handle:
            rendered_image = handle.copy()
        ax.imshow(rendered_image)
        ax.axis("off")
        self.y -= display_height + 0.08
        self.add_paragraph(image.alt, size=SMALL_SIZE)

    def add_line(
        self,
        text: str,
        *,
        size: float,
        indent: float = 0.0,
        weight: str = "normal",
        color: str = "#111111",
    ) -> None:
        line_height = size / 72 * 1.45
        self.ensure_space(line_height + 0.02)
        assert self.fig is not None
        self.fig.text(
            (LEFT_MARGIN + indent) / PAGE_WIDTH,
            self.y / PAGE_HEIGHT,
            text,
            ha="left",
            va="top",
            fontsize=size,
            fontproperties=self.font,
            fontweight=weight,
            color=color,
        )
        self.y -= line_height


def render_markdown(input_path: Path, renderer: PdfRenderer) -> None:
    lines = input_path.read_text(encoding="utf-8").splitlines()
    base_dir = input_path.parent
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            renderer.add_space(0.06)
            index += 1
            continue

        if stripped.startswith("```"):
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            renderer.add_code_block(code_lines)
            index += 1
            continue

        image = parse_image(stripped, base_dir)
        if image is not None:
            renderer.add_image(image)
            index += 1
            continue

        if stripped.startswith("|"):
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            renderer.add_table(table_lines)
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            renderer.add_heading(heading.group(2), len(heading.group(1)))
            index += 1
            continue

        checked = re.match(r"^-\s+\[([ xX])\]\s+(.+)$", stripped)
        if checked:
            renderer.add_bullet(checked.group(2), checked=checked.group(1).lower() == "x")
            index += 1
            continue

        if stripped.startswith("- "):
            renderer.add_bullet(stripped[2:])
            index += 1
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            next_line = lines[index].strip()
            if (
                not next_line
                or next_line.startswith("#")
                or next_line.startswith("- ")
                or next_line.startswith("|")
                or next_line.startswith("```")
                or parse_image(next_line, base_dir) is not None
            ):
                break
            paragraph_lines.append(next_line)
            index += 1
        renderer.add_paragraph(" ".join(paragraph_lines))


def find_cjk_font() -> Path:
    candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/arphic/ukai.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No CJK font found for PDF rendering")


def parse_image(line: str, base_dir: Path) -> ImageRef | None:
    match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
    if not match:
        return None
    path = Path(match.group(2))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return ImageRef(alt=clean_inline(match.group(1)), path=path)


def parse_table(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        cells = [clean_inline(cell.strip()) for cell in line.strip("|").split("|")]
        if all(set(cell.replace(":", "").strip()) <= {"-"} for cell in cells):
            continue
        rows.append(cells)
    return rows


def clean_inline(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    return text.replace("`", "").replace("**", "")


def wrap_text(text: str, width: int) -> list[str]:
    return textwrap.wrap(
        text,
        width=width,
        replace_whitespace=False,
        break_long_words=True,
        break_on_hyphens=False,
    ) or [""]


if __name__ == "__main__":
    main()
