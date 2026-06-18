"""Verify final coursework submission artifacts."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


GITHUB_URL = "https://github.com/ziangbuchu/ML_homework"
UCI_URL = "https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption"
IMPROVED_MODEL = "lstm_transformer_ensemble"
HORIZONS = ("90", "365")


def main() -> None:
    checks = [
        verify_required_files(),
        verify_pdf(),
        verify_report_text(),
        verify_results(),
    ]
    failures = [failure for check in checks for failure in check["failures"]]
    report = {
        "status": "pass" if not failures else "fail",
        "checks": checks,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def verify_required_files() -> dict[str, Any]:
    required = [
        "README.md",
        "environment.yml",
        "configs/data_uci.yaml",
        "scripts/download_uci_power.py",
        "scripts/prepare_data.py",
        "scripts/run_official_experiments.py",
        "scripts/build_ensemble_results.py",
        "scripts/run_naive_baselines.py",
        "scripts/run_ensemble_weight_sweep.py",
        "scripts/summarize_official_results.py",
        "scripts/build_design_report_pdf.py",
        "scripts/build_latex_report.sh",
        "scripts/build_report_pdf.py",
        "reports/ML_homework_report.html",
        "reports/ML_homework_report.tex",
        "reports/ML_homework_report.md",
        "reports/ML_homework_report.pdf",
        "reports/submission_checklist.md",
        "results/official_summary/report_table.md",
        "results/official_summary/metrics_summary.csv",
        "results/official_summary/ensemble_weight_sweep.csv",
        "results/official_summary/ensemble_weight_selection.csv",
        "results/official_summary/ensemble_weight_summary.md",
        "results/official_summary/naive_baselines.csv",
        "results/official_summary/naive_baseline_summary.md",
        "results/official_summary/figures/comparison_h90_seed1.png",
        "results/official_summary/figures/comparison_h365_seed1.png",
    ]
    missing = [path for path in required if not Path(path).is_file()]
    return {
        "name": "required_files",
        "status": "pass" if not missing else "fail",
        "checked": required,
        "failures": [f"missing required file: {path}" for path in missing],
    }


def verify_pdf() -> dict[str, Any]:
    pdf_path = Path("reports/ML_homework_report.pdf")
    failures: list[str] = []
    info: dict[str, str] = {}
    if not pdf_path.is_file():
        failures.append(f"missing PDF: {pdf_path}")
    else:
        try:
            output = subprocess.check_output(
                ["pdfinfo", str(pdf_path)],
                text=True,
                stderr=subprocess.STDOUT,
            )
            info = parse_pdfinfo(output)
        except (OSError, subprocess.CalledProcessError) as error:
            failures.append(f"pdfinfo failed: {error}")
        else:
            pages = int(info.get("Pages", "0"))
            page_size = info.get("Page size", "")
            if pages <= 0:
                failures.append(f"invalid PDF page count: {pages}")
            if "A4" not in page_size:
                failures.append(f"PDF page size is not A4: {page_size}")
            blank_pages = find_blank_text_pages(pdf_path, pages)
            if blank_pages:
                failures.append(f"PDF contains blank text page(s): {blank_pages}")
    return {
        "name": "pdf",
        "status": "pass" if not failures else "fail",
        "pdfinfo": info,
        "failures": failures,
    }


def verify_report_text() -> dict[str, Any]:
    report_paths = [
        Path("reports/ML_homework_report.html"),
        Path("reports/ML_homework_report.tex"),
        Path("reports/ML_homework_report.md"),
    ]
    failures: list[str] = []
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in report_paths
        if path.is_file()
    )
    required_fragments = [
        GITHUB_URL,
        UCI_URL,
        "参考文献",
        "工具辅助说明",
        "LSTM+Transformer Ensemble",
        "实验设计取舍",
        "模型归纳偏置",
        "last_value",
        "验证集调权",
        "test-oracle",
        "有效性边界",
        "最终提交检查",
    ]
    for fragment in required_fragments:
        if fragment not in text:
            failures.append(f"report missing text: {fragment}")
    return {
        "name": "report_text",
        "status": "pass" if not failures else "fail",
        "checked_sources": [str(path) for path in report_paths],
        "required_fragments": required_fragments,
        "failures": failures,
    }


def verify_results() -> dict[str, Any]:
    rows = read_metric_summary(Path("results/official_summary/metrics_summary.csv"))
    failures: list[str] = []
    ranking: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS:
        group = [
            row for row in rows
            if row["split"] == "test" and row["horizon"] == horizon
        ]
        if not group:
            failures.append(f"missing test summary rows for horizon {horizon}")
            continue
        best_mse = min(group, key=lambda row: float(row["mse_mean"]))
        best_mae = min(group, key=lambda row: float(row["mae_mean"]))
        ranking[horizon] = {
            "best_mse_model": best_mse["model"],
            "best_mse": float(best_mse["mse_mean"]),
            "best_mae_model": best_mae["model"],
            "best_mae": float(best_mae["mae_mean"]),
            "model_count": len(group),
        }
        if best_mse["model"] != IMPROVED_MODEL:
            failures.append(f"horizon {horizon} best MSE is {best_mse['model']}, not {IMPROVED_MODEL}")
        if best_mae["model"] != IMPROVED_MODEL:
            failures.append(f"horizon {horizon} best MAE is {best_mae['model']}, not {IMPROVED_MODEL}")
    return {
        "name": "official_results",
        "status": "pass" if not failures else "fail",
        "ranking": ranking,
        "failures": failures,
    }


def parse_pdfinfo(output: str) -> dict[str, str]:
    info: dict[str, str] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        info[key.strip()] = value.strip()
    return info


def find_blank_text_pages(pdf_path: Path, pages: int) -> list[int]:
    if not shutil.which("pdftotext"):
        return []
    blank_pages: list[int] = []
    for page in range(1, pages + 1):
        output = subprocess.check_output(
            ["pdftotext", "-f", str(page), "-l", str(page), str(pdf_path), "-"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        if not output.replace("\f", "").strip():
            blank_pages.append(page)
    return blank_pages


def read_metric_summary(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing metric summary: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
