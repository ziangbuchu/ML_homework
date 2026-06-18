# 完善作业最终提交质量

## Goal

在现有代码、实验和 PDF 报告已经完成的基础上，补齐提交前最有价值的 polish：可复现说明、自动提交自检、最终清单证据和 README 导航，降低助教复查和提交遗漏风险。

## What I Already Know

- 作业要求来自 `docs/assignment-requirements.md`，必须提交 PDF、完整可运行代码、GitHub 链接、结果截图、参考文献和 MSE/MAE mean/std。
- 当前报告已存在：`reports/ML_homework_report.pdf`，`pdfinfo` 显示 5 页 A4。
- 当前最终改进模型为 `LSTM+TF Ensemble`，在 90/365 两个 horizon 上 test MSE 和 test MAE 都是第一。
- 当前结果汇总在 `results/official_summary/`，正式汇总脚本输出 `runs=120`、`summary_rows=24`。
- 当前 README 有基本运行命令，但缺一个明确的“提交前验收/一键自检”入口。
- 当前 submission checklist 是静态勾选，缺少自动化证据输出。

## Requirements

- 新增一个提交自检脚本，自动验证：
  - PDF 存在、可由 `pdfinfo` 解析、A4、页数大于 0。
  - 报告源稿包含 GitHub 链接、UCI 链接、参考文献、工具辅助说明。
  - official summary 表存在，并且 `lstm_transformer_ensemble` 在 90/365 的 test MSE 上排名第一。
  - 90/365 对比图存在。
  - 关键代码/脚本存在。
- 新增复现说明文档，面向助教说明从环境、数据、训练、ensemble、汇总、报告到提交自检的命令。
- 更新 README 和提交 checklist，加入 final verification 命令与最新 improved result。
- 不重跑所有训练；只做轻量自检、文档和报告更新。
- 不删除旧结果或未提交文件。

## Acceptance Criteria

- [x] 新增 `scripts/verify_submission.py` 并能输出结构化通过结果。
- [x] 新增 `docs/reproducibility.md`，覆盖完整复现路径和当前结果边界。
- [x] README 包含 final verification 命令和最新结果摘要。
- [x] `reports/submission_checklist.md` 包含自动自检命令与结果证据。
- [x] 报告源稿/PDF 补充单人/工具辅助/提交自检相关说明。
- [x] `PYTHONPATH=src pytest` 通过。
- [x] `PYTHONPATH=src python scripts/verify_submission.py` 通过。
- [x] PDF 重新生成且 `pdfinfo` 可解析。

## Completion Evidence

- 新增自检脚本：`scripts/verify_submission.py`。
  - 验证 PDF、报告文本、关键文件、结果图、official summary 排名。
  - 输出 JSON，全部通过时 `status=pass`。
- 新增复现文档：`docs/reproducibility.md`。
- README 已加入复现文档入口、最终结果摘要和提交前自检命令。
- `reports/submission_checklist.md` 已加入 improved model 排名、自检命令和复现说明。
- `reports/ML_homework_report.md` 已加入单人说明、复现与自检说明；`reports/ML_homework_report.pdf` 已重建。
- PDF 版面抽查：第 5 页渲染正常，新增复现/自检说明未溢出。
- 验证命令：
  - `PYTHONPATH=src pytest`：13 passed。
  - `PYTHONPATH=src python scripts/verify_submission.py`：`status=pass`。
  - `pdfinfo reports/ML_homework_report.pdf`：5 页 A4、未加密、可解析。
- Spec update：`.trellis/spec/backend/quality-guidelines.md` 记录 `scripts/verify_submission.py` 的命令契约。

## Out of Scope

- 不重新下载数据。
- 不重跑 30 个正式训练 run。
- 不进一步调模型结构。
- 不进行 git commit、push 或删除数据。

## Technical Notes

- 主要涉及 `scripts/`、`docs/`、`README.md`、`reports/`。
- 结果可信边界：指标为标准化目标尺度；ensemble 使用固定 0.5/0.5 权重，不是测试集调权。
