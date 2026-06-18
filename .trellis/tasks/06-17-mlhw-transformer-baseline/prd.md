# 实现 Transformer 短期与长期预测基线

## Goal

实现课程必做的 Transformer baseline，并在统一训练评估框架中支持 90 天和 365 天预测窗口。

## Requirements

- 模型输入为 `batch_size x 90 x num_features`。
- 使用位置编码和 Transformer encoder 建模时间序列。
- 模型输出为 `batch_size x output_length`，其中 `output_length` 为 90 或 365。
- 短期和长期配置、训练产物和参数彼此独立。
- 与 LSTM 使用同一数据、训练、评估和保存协议。

## Acceptance Criteria

- [x] Transformer 模型能在短期配置下完成 smoke 训练和评估。
- [x] Transformer 模型能在长期配置下完成 smoke 训练和评估。
- [x] 产物路径、指标和曲线图与 LSTM baseline 可直接横向比较。
- [x] 模型说明可直接支撑报告中的 Transformer 方法小节。

## Dependencies

- 依赖 `06-17-mlhw-data-pipeline`。
- 依赖 `06-17-mlhw-training-eval-framework`。

## Decision Points

- Transformer 层数、head 数、embedding 维度、dropout 和训练 epoch 的 smoke 默认值：已决定 `d_model=64`、`nhead=4`、`num_layers=2`、`dim_feedforward=128`、`dropout=0.1`、smoke `epochs=2`。
- 输出头是 pooled representation 投影，还是 flatten 后投影：已决定 mean pooling + horizon projection。

## Out of Scope

- 不运行正式 5 seed 结果。
- 不把 Transformer baseline 和改进模型混在同一个实现里。

## Verification

- `PYTHONPATH=src pytest`：8 passed。
- `PYTHONPATH=src python scripts/run_experiment.py --config configs/train_transformer_smoke_90.yaml`
  - run dir：`results/runs/smoke_transformer_output90/seed_42`
  - test MSE：`0.4549692939000307`
  - test MAE：`0.5309977374151155`
  - prediction/target shape：`(32, 90)`
- `PYTHONPATH=src python scripts/run_experiment.py --config configs/train_transformer_smoke_365.yaml`
  - run dir：`results/runs/smoke_transformer_output365/seed_42`
  - test MSE：`0.7479247378129837`
  - test MAE：`0.6891363323252622`
  - prediction/target shape：`(32, 365)`
