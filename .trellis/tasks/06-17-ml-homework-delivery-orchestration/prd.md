# ML 大作业总控交付

## Goal

统筹完成 2026 年专硕机器学习课程大作业：以家庭电力消耗多变量时间序列为对象，交付完整可运行代码、三类模型实验、结果图表和 PDF 报告。

## What I Already Know

- 截止时间是 2026 年 7 月 15 日中午 12:00 之前。
- 最终需要提交 PDF 报告，并在报告中附 GitHub 链接。
- 任务是基于过去 90 天的多变量日级数据预测未来每日 `global_active_power`。
- 必须分别覆盖短期预测 90 天和长期预测 365 天，且短期/长期模型需要独立训练。
- 必做方法是 LSTM、Transformer、自提出改进模型。
- 每种方法至少五轮实验，指标为 MSE、MAE，并报告 mean/std。
- 报告需要包含预测曲线与 `Ground Truth` 对比截图和参考文献。
- 用户希望使用新的 conda 环境，并在需要决策的位置进行 grill。

## Subtasks

1. [`06-17-mlhw-env-scaffold`](../06-17-mlhw-env-scaffold/prd.md) - 搭建独立 conda 环境与项目骨架。
2. [`06-17-mlhw-data-pipeline`](../06-17-mlhw-data-pipeline/prd.md) - 实现数据获取、清洗、日级聚合与窗口构造。
3. [`06-17-mlhw-training-eval-framework`](../06-17-mlhw-training-eval-framework/prd.md) - 实现统一训练、评估、日志、绘图和结果保存框架。
4. [`06-17-mlhw-lstm-baseline`](../06-17-mlhw-lstm-baseline/prd.md) - 实现 LSTM 短期/长期预测 baseline。
5. [`06-17-mlhw-transformer-baseline`](../06-17-mlhw-transformer-baseline/prd.md) - 实现 Transformer 短期/长期预测 baseline。
6. [`06-17-mlhw-improved-model`](../06-17-mlhw-improved-model/prd.md) - 设计并实现自提出改进模型。
7. [`06-17-mlhw-official-experiments`](../06-17-mlhw-official-experiments/prd.md) - 运行正式五轮实验并生成结果图表。
8. [`06-17-mlhw-report-submission`](../06-17-mlhw-report-submission/prd.md) - 撰写 PDF 报告并整理最终提交材料。

## Recommended Execution Order

```text
env-scaffold
  -> data-pipeline
  -> training-eval-framework
  -> lstm-baseline
  -> transformer-baseline
  -> improved-model
  -> official-experiments
  -> report-submission
```

模型实现 task 可以在训练评估框架完成后并行推进；正式实验必须等三类模型和数据流水线都稳定后再启动。

## Decision Gates

- **环境 gate**: conda 环境名、Python 版本、CPU/GPU 与 PyTorch 安装方式。
- **数据 gate**: 课程数据文件真实路径和测试集文件名是 `tes.csv` 还是 `test.csv`。
- **特征 gate**: 是否融合天气数据；默认先不引入天气，除非数据可得且处理成本低。
- **改进模型 gate**: 自提出模型方向需要 grill 后确认，避免实现了难解释或报告不可写的结构。
- **实验 gate**: 正式五轮实验前，需要确认 smoke 结果有效、日志/图表/表格产物完整。
- **报告 gate**: 是否单人/组队、是否需要注明 ChatGPT/DeepSeek 辅助，以及最终 PDF 制作方式。

## Acceptance Criteria

- [x] 8 个子任务均有 `prd.md`，且挂在本总任务下。
- [x] 每个子任务有清晰的目标、依赖、验收标准和决策点。
- [x] 后续可以按子任务逐个进入 Trellis Phase 2 执行，而不用重新拆分项目。
- [x] 总任务最终能追踪到完整代码、实验结果、图表、报告和提交检查证据。

## Completion Evidence

- 环境：`environment.yml` 创建 `ml-homework` conda 环境，`scripts/smoke_env.py` 验证 PyTorch/CUDA。
- 数据：`configs/data_uci.yaml`、`scripts/download_uci_power.py`、`scripts/prepare_data.py` 和 `results/data/uci_power_summary.json`。
- 模型与训练：`src/ml_homework/`、`scripts/run_experiment.py`、`configs/train_*`。
- 正式实验：`scripts/run_official_experiments.py`、`configs/official/`、`results/official_summary/`。
- 报告提交物：`reports/ML_homework_report.md`、`reports/ML_homework_report.pdf`、`reports/submission_checklist.md`。
- 验证：`PYTHONPATH=src pytest` 通过 9 项；正式汇总脚本输出 90 条 run/split 指标和 18 条汇总行；PDF 为 5 页 A4。

## Out of Scope

- 本总控 task 不直接实现代码。
- 本总控 task 不直接下载数据、训练模型或生成 PDF。
- 本总控 task 不替代各子任务的质量检查。

## Technical Notes

- 需求来源：`docs/assignment-requirements.md`、`docs/project-foundation.md`。
- `.trellis/tasks/00-bootstrap-guidelines` 仍为历史 bootstrap task，和本作业交付任务树并行存在。
- 报告中的强结论仅限当前 8 epoch、5 seed、标准化指标正式实验结果；不声称改进模型优于 baseline。
