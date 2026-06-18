# 撰写 PDF 报告并整理最终提交材料

## Goal

按课程要求完成 PDF 报告和最终提交检查，确保代码、GitHub 链接、截图、参考文献和工具辅助说明齐全。

## Requirements

- 报告结构必须包含四部分：
  1. 问题介绍
  2. 模型，可以包含少量伪代码
  3. 结果与分析
  4. 讨论
- 报告中附 GitHub 链接：`https://github.com/ziangbuchu/ML_homework`。
- 嵌入结果截图或图片，包括预测曲线与 `Ground Truth` 对比。
- 包含三类模型在短期/长期预测上的 MSE、MAE mean/std 表。
- 明确参考文献。
- 如使用 ChatGPT、DeepSeek 等工具辅助报告撰写，需要按课程要求注明。
- 最终检查提交入口、截止时间和 PDF 可打开性。

## Acceptance Criteria

- [x] PDF 报告存在且能打开。
- [x] 报告四部分结构完整。
- [x] 报告包含 GitHub 链接、结果图、mean/std 表和参考文献。
- [x] 结果表中的数字能追溯到 official experiments 产物。
- [x] 最终提交检查清单完成。

## Completion Evidence

- 报告源稿：`reports/ML_homework_report.md`。
- PDF 报告：`reports/ML_homework_report.pdf`，`pdfinfo` 显示 5 页 A4、未加密、可解析。
- PDF 生成脚本：`scripts/build_report_pdf.py`，命令为 `PYTHONPATH=src python scripts/build_report_pdf.py --input reports/ML_homework_report.md --output reports/ML_homework_report.pdf`。
- 结果表来源：`results/official_summary/metrics_summary.csv` 和 `results/official_summary/report_table.md`。
- 结果图来源：`results/official_summary/figures/comparison_h90_seed1.png`、`results/official_summary/figures/comparison_h365_seed1.png`。
- 提交清单：`reports/submission_checklist.md`。
- 验证：`PYTHONPATH=src pytest` 通过 9 项；`scripts/summarize_official_results.py` 输出 `{"runs": 90, "summary_rows": 18}`。

## Dependencies

- 依赖 `06-17-mlhw-official-experiments`。

## Decision Points

- 报告制作方式：Markdown/Quarto、LaTeX、Word/WPS；推荐选择最快能稳定导出 PDF 的方式。
- 是否组队；若组队，需要列明作者贡献和研究领域。
- 工具辅助声明措辞。

## Out of Scope

- 不重新训练模型。
- 不在报告阶段改动实验数据或指标口径。

## Technical Notes

- 提交截止：2026 年 7 月 15 日中午 12:00 之前。
- 提交链接：`https://docs.qq.com/form/page/DT3pqV3pNcGV6TG1z`。
