# 重新设计并美化 PDF 报告

## Goal

把当前观感不足的 PDF 报告替换为更像正式数据实验档案的设计版 A4 PDF。最新用户反馈是“PDF 太丑，用 frontend-design 重新设计”，因此最终主路径改为 HTML/CSS 设计稿经 Chrome headless 导出 PDF；LaTeX 版本保留为 fallback。

## What I Already Know

- 当前 PDF 内容完整但版面朴素，用户明确认为“太难看”。
- 系统 PATH 中没有 `xelatex`、`pdflatex`、`latexmk` 或 `tectonic`。
- conda 可用，现有环境为 `ml-homework`。
- 系统字体存在 `Noto Sans CJK SC`、`Noto Serif CJK SC` 和 `DejaVu Sans Mono`，可以支持中文 LaTeX。
- 当前报告正文源稿是 `reports/ML_homework_report.md`，最终 PDF 路径是 `reports/ML_homework_report.pdf`。
- 本机存在 `/usr/bin/google-chrome`，可用于 headless PDF 导出。

## Requirements

- 用 `frontend-design` 思路重做报告视觉：蓝色数据档案风格、强层级、清晰表格、可读图页，避免绿色主题和朴素默认 PDF。
- 新增 HTML/CSS 设计源文件：`reports/ML_homework_report.html`。
- 新增默认 PDF 构建脚本：`scripts/build_design_report_pdf.py`，输出仍为 `reports/ML_homework_report.pdf`。
- 构建脚本需要自动处理 Chrome 打印偶发空白页，避免提交 PDF 混入空白页。
- 用 LaTeX 编写更美观的报告源文件：`reports/ML_homework_report.tex`。
- 使用用户态工具安装 LaTeX 编译器；优先安装 `tectonic` 到 `ml-homework` conda 环境，不使用 `sudo`。
- LaTeX PDF 必须包含现有报告的关键内容：
  - GitHub 链接、UCI 数据源、问题介绍、模型、结果与分析、讨论。
  - 结果表、90/365 对比图、参考文献、工具辅助说明、复现与提交自检说明。
  - 最终 `LSTM+TF Ensemble` 最优结论和指标。
- 更新 README、复现说明和自检脚本，使最终报告构建以 HTML/CSS 设计版为主，LaTeX 为 fallback。
- 保留旧 `scripts/build_report_pdf.py` 作为 fallback，不删除。
- 根据用户追加要求，补充体现思考深度的模型/实验完善：
  - 报告解释实验设计取舍、模型归纳偏置、负结果和有效性边界。
  - 新增 last-value 与 input-mean naive baseline，证明神经模型确实超过简单持久性预测。
  - 新增 LSTM+Transformer ensemble 验证集调权消融，检验固定 `0.5/0.5` 权重是否稳健。
  - 复现说明和自检脚本覆盖新增 ablation 产物。

## Acceptance Criteria

- [x] 用户态安装的 LaTeX 编译工具可用。
- [x] `reports/ML_homework_report.tex` 存在且可编译。
- [x] `reports/ML_homework_report.html` 存在，并作为默认设计版 PDF 源文件。
- [x] `reports/ML_homework_report.pdf` 由 HTML/CSS 设计版构建脚本生成，版面优于旧 matplotlib 和旧 LaTeX 版本。
- [x] PDF 为 A4、可由 `pdfinfo` 解析，字体正常显示中文和图表。
- [x] README/复现说明包含设计版 PDF 构建命令和 LaTeX fallback。
- [x] `scripts/verify_submission.py` 检查 HTML 源文件、设计版构建脚本、最终 PDF 和空白页。
- [x] `scripts/run_ensemble_weight_sweep.py` 生成 ensemble 权重消融结果。
- [x] `scripts/run_naive_baselines.py` 生成 deterministic naive baseline 结果。
- [x] 报告包含实验设计取舍、模型归纳偏置、验证集调权消融和有效性边界。
- [x] `PYTHONPATH=src pytest` 通过。
- [x] `PYTHONPATH=src python scripts/verify_submission.py` 通过。

## Completion Evidence

- 本地用户态编译器：`.tools/tectonic/tectonic`，版本 `Tectonic 0.16.9`。
- 构建命令：`bash scripts/build_latex_report.sh`。
- 最终 PDF：`reports/ML_homework_report.pdf`，`pdfinfo` 显示 6 页 A4。
- `pdffonts reports/ML_homework_report.pdf` 显示所有字体均 embedded/subset。
- 页面预览检查过首页、目录、结果表、两张曲线图和结尾检查页，无明显裁切或重叠。
- conda 环境验证：`/data1/lf/miniconda3/bin/conda run -n ml-homework bash -lc 'PYTHONPATH=src pytest && PYTHONPATH=src python scripts/verify_submission.py'` 通过，13 tests passed，submission check `status: pass`。
- 追加模型/实验完善后，`PYTHONPATH=src python scripts/run_ensemble_weight_sweep.py --device cpu` 生成 220 行 sweep 和 10 行 selection。
- 新增 ablation 产物：
  - `results/official_summary/ensemble_weight_sweep.csv`
  - `results/official_summary/ensemble_weight_selection.csv`
  - `results/official_summary/ensemble_weight_summary.md`
- 新增 naive baseline 产物：
  - `results/official_summary/naive_baselines.csv`
  - `results/official_summary/naive_baseline_summary.md`
- Naive baseline test 结果：
  - `input_mean`: 90 天 MSE/MAE `0.736495` / `0.711796`，365 天 `1.036502` / `0.817853`。
  - `last_value`: 90 天 MSE/MAE `0.923572` / `0.757890`，365 天 `1.431294` / `0.960546`。
  - 二者均明显弱于 LSTM+Transformer Ensemble，支持神经模型确实超过简单持久性预测。
- 关键 ablation 结论：
  - 90 天：equal `0.5` test MSE/MAE 为 `0.374842±0.021491` / `0.476893±0.012027`，validation-selected 变差为 `0.446209±0.043318` / `0.518641±0.027420`。
  - 365 天：validation-selected 略优于 equal `0.5`，test MSE/MAE 为 `0.442416±0.017311` / `0.513660±0.010671`。
  - 综合两个 horizon，最终仍保留固定 equal ensemble，因为它不依赖测试集调参，并且在 90 天预测上更稳健。
- 最新 conda 端到端验证：`/data1/lf/miniconda3/bin/conda run -n ml-homework bash -lc 'PYTHONPATH=src python scripts/run_naive_baselines.py && PYTHONPATH=src python scripts/run_ensemble_weight_sweep.py --device cpu && bash scripts/build_latex_report.sh && PYTHONPATH=src pytest && PYTHONPATH=src python scripts/verify_submission.py'` 通过，PDF 为 8 页 A4，13 tests passed，submission check `status: pass`。
- 设计版报告源文件：`reports/ML_homework_report.html`。
- 设计版构建命令：`python scripts/build_design_report_pdf.py`。
- 构建脚本会调用 Chrome 生成 PDF，并用 `pdftotext` 检测空白页；当前生成时剔除 Chrome 产生的第 2 页空白页。
- 当前设计方向：蓝色数据实验档案，封面指标卡、网格纸背景、结果表/消融表/曲线页/提交自检页均独立排版。
- 最终 conda 验证：
  - `/data1/lf/miniconda3/bin/conda run -n ml-homework python scripts/build_design_report_pdf.py` 通过，生成 8 页 A4 PDF。
  - `/data1/lf/miniconda3/bin/conda run -n ml-homework env PYTHONPATH=src pytest` 通过，13 tests passed。
  - `/data1/lf/miniconda3/bin/conda run -n ml-homework env PYTHONPATH=src python scripts/verify_submission.py` 通过，submission check `status: pass`。

## Out of Scope

- 不重跑正式训练。
- 不更改实验结果和指标口径。
- 不删除旧 Markdown 源稿或旧 Python PDF 构建脚本。
- 不使用系统级 `sudo apt install`。

## Technical Notes

- 优先尝试 `conda install -n ml-homework -c conda-forge tectonic`。
- LaTeX 推荐使用 `fontspec`、`xeCJK`、`booktabs`、`tabularx`、`graphicx`、`hyperref`。
- 中文字体优先：`Noto Serif CJK SC` 正文，`Noto Sans CJK SC` 标题/强调。
