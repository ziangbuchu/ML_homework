# 运行正式五轮实验并生成结果图表

## Goal

在三类模型和两个预测窗口上运行正式实验，生成课程报告需要的 mean/std 表格、预测曲线和截图素材。

## Requirements

- 覆盖 3 类模型：LSTM、Transformer、改进模型。
- 覆盖 2 个预测窗口：90 天、365 天。
- 每个配置至少运行 5 个 seed。
- 每次运行保存 config、seed、metrics、prediction、ground truth、curve figure。
- 汇总 MSE、MAE 的 mean/std。
- 生成短期和长期预测曲线与 `Ground Truth` 对比图。
- 输出报告可直接引用的表格和图像路径清单。

## Acceptance Criteria

- [x] 至少完成 `3 models x 2 horizons x 5 seeds` 的正式实验矩阵，或明确记录无法完成的配置和原因。
- [x] 生成 `MSE`、`MAE` 的 mean/std 汇总表。
- [x] 生成短期、长期预测曲线图和可贴入报告的截图/图片。
- [x] 汇总文件中能追溯每个结果来自哪个 config、seed 和产物目录。
- [x] 发现异常结果时有诊断记录，而不是静默丢弃。

## Completion Evidence

- 正式配置：`configs/official/` 下 30 个 YAML，覆盖 3 类模型、2 个 horizon、5 个 seed。
- 正式运行：`results/official_runs/` 下 30 个 `metrics.json`，运行命令为 `PYTHONPATH=src python scripts/run_official_experiments.py --epochs 8 --batch-size 32 --device cpu`。
- 汇总命令：`PYTHONPATH=src python scripts/summarize_official_results.py`，输出 `{"runs": 90, "summary_rows": 18}`。
- 汇总产物：`results/official_summary/metrics_by_run.csv`、`metrics_summary.csv`、`report_table.md`、`artifact_manifest.json`。
- 报告图：`results/official_summary/figures/comparison_h90_seed1.png`、`results/official_summary/figures/comparison_h365_seed1.png`。
- 结论边界：本轮 CPU 正式实验使用 8 epochs，指标为标准化目标尺度上的 MSE/MAE；LSTM 在 90 天测试集最好，Transformer 在 365 天测试集最好，改进模型未超过两类 baseline。

## Dependencies

- 依赖 `06-17-mlhw-lstm-baseline`。
- 依赖 `06-17-mlhw-transformer-baseline`。
- 依赖 `06-17-mlhw-improved-model`。

## Decision Points

- 每个正式 run 的 epoch、batch size 和 early stopping 设置。
- 如果 compute 不足，是降 epoch 做课程可交付，还是延长训练追求更好结果。
- 是否对异常 seed 复跑；默认不删结果，先记录原因。

## Out of Scope

- 不改变模型结构。
- 不在结果不理想时临时改指标口径。

## Technical Notes

- 这是最容易 overclaim 的阶段：报告只能声称当前实验矩阵支持的结论。
