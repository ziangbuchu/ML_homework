# Quality Guidelines

> Code quality standards for backend development.

---

## Overview

<!--
Document your project's quality standards here.

Questions to answer:
- What patterns are forbidden?
- What linting rules do you enforce?
- What are your testing requirements?
- What code review standards apply?
-->

(To be filled by the team)

---

## Forbidden Patterns

<!-- Patterns that should never be used and why -->

(To be filled by the team)

---

## Required Patterns

<!-- Patterns that must always be used -->

(To be filled by the team)

---

## Testing Requirements

### Submission Verification Command

The coursework project keeps a lightweight final verification command:

```bash
PYTHONPATH=src python scripts/verify_submission.py
```

Contract:

- It takes no required arguments.
- It prints a JSON object with top-level `status` and per-check `checks`.
- It exits with code `0` only when all submission checks pass.
- It exits with code `1` when any required file, PDF property, report text fragment, result ranking, or figure artifact is missing/invalid.

Use this command after regenerating reports or official summaries so README, PDF, result tables, and final submission artifacts do not drift.

### Designed Report Build Command

#### 1. Scope / Trigger

- Trigger: final coursework report regeneration or PDF layout changes.
- Scope: build the submission PDF from `reports/ML_homework_report.html` without rerunning model training.

#### 2. Signatures

```bash
python scripts/build_design_report_pdf.py
```

Optional override:

```bash
python scripts/build_design_report_pdf.py --chrome /path/to/google-chrome
```

#### 3. Contracts

- Input source: `reports/ML_homework_report.html`.
- Output PDF: `reports/ML_homework_report.pdf`.
- Preferred renderer: Chrome/Chromium, resolved from `--chrome`, `google-chrome`, `google-chrome-stable`, `chromium`, or `chromium-browser`.
- The script must fail nonzero if the renderer is missing, the HTML source is missing, the PDF is not created, or `pdfinfo` cannot parse the generated PDF when `pdfinfo` is available.
- If Chrome creates blank text pages, the script may use `pdftotext`, `pdfseparate`, and `pdfunite` to remove only pages with no extracted text.
- `bash scripts/build_latex_report.sh` and `scripts/build_report_pdf.py` remain fallback paths only; final submission should use the designed HTML/CSS PDF.

#### 4. Validation & Error Matrix

| Condition | Expected behavior |
| --- | --- |
| No Chrome/Chromium executable found | Raise `FileNotFoundError` |
| Missing `reports/ML_homework_report.html` | Raise `FileNotFoundError` |
| Chrome PDF export fails | Propagate the renderer exit code |
| PDF file missing or empty | Print expected PDF path and exit `1` |
| `pdfinfo` cannot parse the PDF | Propagate `pdfinfo` failure |
| Blank text pages detected and Poppler tools available | Rebuild final PDF without those pages and print removed page numbers |

#### 5. Good/Base/Bad Cases

- Good: `python scripts/build_design_report_pdf.py` produces an A4 PDF with visible figures, blue theme, no blank pages, and readable Chinese text.
- Base: `--chrome` points to a working external Chrome/Chromium binary and produces the same output path.
- Bad: using the legacy Python PDF builder or LaTeX fallback as the primary submission artifact after the designed HTML report is available.

#### 6. Tests Required

- Run `python scripts/build_design_report_pdf.py`.
- Run `pdfinfo reports/ML_homework_report.pdf` and confirm A4 page size with positive page count.
- Render representative pages with `pdftoppm` and visually check cover, result tables, figure page, and final checklist.
- Run `PYTHONPATH=src python scripts/verify_submission.py`.

#### 7. Wrong vs Correct

Wrong:

```bash
PYTHONPATH=src python scripts/build_report_pdf.py --input reports/ML_homework_report.md --output reports/ML_homework_report.pdf
PYTHONPATH=src python scripts/verify_submission.py
```

Correct:

```bash
python scripts/build_design_report_pdf.py
PYTHONPATH=src python scripts/verify_submission.py
```

(To be filled by the team)

---

## Code Review Checklist

<!-- What reviewers should check -->

(To be filled by the team)
