# 实现 LSTM 短期与长期预测基线

## Goal

实现课程必做的 LSTM baseline，并在统一训练评估框架中支持 90 天和 365 天预测窗口。

## Requirements

- 模型输入为 `batch_size x 90 x num_features`。
- 模型输出为 `batch_size x output_length`，其中 `output_length` 为 90 或 365。
- 短期和长期模型配置、训练产物和参数彼此独立。
- 使用统一数据流水线、训练循环、指标计算和结果保存格式。
- 提供至少一个快速 smoke 配置，正式五轮实验留给 official experiments task。

## Acceptance Criteria

- [x] LSTM 模型能在短期配置下完成 smoke 训练和评估。
- [x] LSTM 模型能在长期配置下完成 smoke 训练和评估。
- [x] 保存的指标、预测数组和曲线图格式与训练评估框架一致。
- [x] README 或模型说明中有 LSTM baseline 的方法描述，便于报告复用。

## Dependencies

- 依赖 `06-17-mlhw-data-pipeline`。
- 依赖 `06-17-mlhw-training-eval-framework`。

## Decision Points

- LSTM 层数、hidden size、dropout 和训练 epoch 的 smoke 默认值：已决定 `hidden_size=64`、`num_layers=1`、`dropout=0.0`、smoke `epochs=2`。
- 是否使用 encoder-only 最后 hidden state 直接投影，或 seq2seq 解码结构：已决定 encoder-only LSTM + projection。

## Out of Scope

- 不运行正式 5 seed 结果。
- 不调参追求最优性能。

## Verification

- `PYTHONPATH=src pytest`：7 passed。
- `PYTHONPATH=src python scripts/run_experiment.py --config configs/train_lstm_smoke_90.yaml`
  - run dir：`results/runs/smoke_lstm_output90/seed_42`
  - test MSE：`0.2616983067511876`
  - test MAE：`0.39919169794014425`
  - prediction/target shape：`(32, 90)`
- `PYTHONPATH=src python scripts/run_experiment.py --config configs/train_lstm_smoke_365.yaml`
  - run dir：`results/runs/smoke_lstm_output365/seed_42`
  - test MSE：`0.6223550638883156`
  - test MAE：`0.6274952109752395`
  - prediction/target shape：`(32, 365)`
