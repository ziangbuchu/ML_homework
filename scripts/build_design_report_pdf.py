"""Build the designed HTML coursework report into a PDF using Chrome."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/ML_homework_report.html")
    parser.add_argument("--output", default="reports/ML_homework_report.pdf")
    parser.add_argument("--chrome", default=None)
    args = parser.parse_args()

    repo_root = Path.cwd()
    input_path = resolve(repo_root, args.input)
    output_path = resolve(repo_root, args.output)
    chrome = args.chrome or find_chrome()

    if not input_path.is_file():
        raise FileNotFoundError(f"missing HTML report: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=1000",
        "--no-pdf-header-footer",
        f"--print-to-pdf={output_path}",
        input_path.as_uri(),
    ]
    subprocess.run(command, check=True)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"PDF was not created: {output_path}")
    removed_pages = strip_blank_pages(output_path)
    if shutil.which("pdfinfo"):
        subprocess.run(["pdfinfo", str(output_path)], check=True, stdout=subprocess.DEVNULL)
    if removed_pages:
        joined = ", ".join(str(page) for page in removed_pages)
        print(f"Removed blank PDF page(s): {joined}")
    print(f"Built {output_path}")


def resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def find_chrome() -> str:
    candidates = [
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
    ]
    for candidate in candidates:
        path = shutil.which(candidate)
        if path:
            return path
    raise FileNotFoundError("Chrome/Chromium executable not found")


def strip_blank_pages(pdf_path: Path) -> list[int]:
    if not all(shutil.which(tool) for tool in ["pdfinfo", "pdftotext", "pdfseparate", "pdfunite"]):
        return []

    page_count = get_page_count(pdf_path)
    blank_pages = [
        page
        for page in range(1, page_count + 1)
        if not page_text(pdf_path, page).strip()
    ]
    if not blank_pages:
        return []
    keep_pages = [page for page in range(1, page_count + 1) if page not in set(blank_pages)]
    if not keep_pages:
        raise RuntimeError("all generated PDF pages look blank")

    with tempfile.TemporaryDirectory(prefix="mlhw_report_pdf_") as tmp:
        tmp_path = Path(tmp)
        page_pattern = tmp_path / "page-%03d.pdf"
        subprocess.run(
            ["pdfseparate", str(pdf_path), str(page_pattern)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        kept_files = [str(tmp_path / f"page-{page:03d}.pdf") for page in keep_pages]
        cleaned_path = tmp_path / "cleaned.pdf"
        subprocess.run(
            ["pdfunite", *kept_files, str(cleaned_path)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        same_dir_tmp = pdf_path.with_suffix(".cleaned.tmp.pdf")
        shutil.copyfile(cleaned_path, same_dir_tmp)
        same_dir_tmp.replace(pdf_path)
    return blank_pages


def get_page_count(pdf_path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"pdfinfo did not report page count for {pdf_path}")


def page_text(pdf_path: Path, page: int) -> str:
    result = subprocess.run(
        ["pdftotext", "-f", str(page), "-l", str(page), str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.replace("\f", "").strip()


if __name__ == "__main__":
    main()
