# 实现统一训练评估与结果记录框架

## Goal

建立所有模型共享的训练、评估、配置、日志、预测保存和绘图框架，避免每个模型各写一套指标逻辑。

## Requirements

- 提供统一配置方式，覆盖模型名、预测窗口、seed、训练参数、数据路径和输出目录。
- 提供共享训练循环，支持 early stopping 或固定 epoch，并清楚记录训练过程。
- 实现 MSE、MAE 指标，保证所有模型使用同一计算口径。
- 固定随机种子，记录运行配置和 seed。
- 保存每次运行的指标、预测数组、Ground Truth 数组和曲线图。
- 提供命令行入口用于 train/eval/plot 或一体化运行。
- 结果目录结构能支撑后续 3 类模型 x 2 horizons x 5 seeds 的汇总。

## Acceptance Criteria

- [x] 一个 dummy 或极小真实样本 smoke 能完整走完 train/eval/save/plot。
- [x] 指标计算有单元测试或可验证的小样本测试。
- [x] 运行产物包含 config、metrics、predictions、figures。
- [x] LSTM、Transformer、改进模型可以通过同一接口接入。

## Dependencies

- 依赖 `06-17-mlhw-env-scaffold`。
- 依赖或并行对接 `06-17-mlhw-data-pipeline` 的 Dataset 接口。

## Decision Points

- 配置格式：已决定使用 YAML，便于报告和复现实验引用。
- 是否引入实验追踪工具：已决定不用 WandB 等外部服务，先用本地 JSON/CSV/PNG。
- 保存粒度：正式实验必须保存每个 seed 的原始指标和预测曲线。

## Out of Scope

- 不承诺具体模型达到某个精度。
- 不负责最终五轮实验汇总。

## Technical Notes

- 该 task 是后续三个模型 task 的共享依赖。
- 重点防止指标口径、seed 控制、结果路径在不同模型之间分叉。
- 共享入口：`PYTHONPATH=src python scripts/run_experiment.py --config configs/train_smoke.yaml`。
- 当前只内置 `linear` smoke 模型；后续 LSTM/Transformer/改进模型通过 `src/ml_homework/models.py` 的 registry 接入。
- UCI 小样本 smoke 验证：
  - run dir：`results/runs/smoke_linear_output90/seed_42/`
  - metrics：test MSE `0.6471521715564199`，test MAE `0.6406481783397289`
  - predictions：`predictions_test.npz` 中 prediction/target 形状均为 `(32, 90)`
  - artifacts：`config.yaml`、`metrics.json`、`history.csv`、`predictions_test.npz`、`model.pt`、`figures/prediction_*.png`
