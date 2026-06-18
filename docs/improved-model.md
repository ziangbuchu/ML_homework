# Improved Model: LSTM + Transformer Ensemble

## 动机

家庭电力消耗同时包含短期局部波动和较长时间范围的上下文模式。正式实验中，LSTM 在 90 天短期预测上更稳，Transformer 在 365 天长期预测上更强。最终改进模型利用这一互补性，将两类 baseline 的预测做固定权重 ensemble。

在最终提交版之前，也尝试过 TCN+Transformer：先用 temporal convolution block 提取局部波动，再接 Transformer encoder。但正式五轮结果显示该结构没有超过 baseline，因此保留为负结果/对照，不作为最终改进模型结论。

## 结构

最终 improved model 使用 equal-weight ensemble：

```text
y_lstm = LSTM(X)
y_transformer = Transformer(X)
y_ensemble = 0.5 * y_lstm + 0.5 * y_transformer
```

权重固定为 `0.5/0.5`，没有按测试集调权。生成脚本会加载正式 LSTM 与 Transformer run 的 `model.pt`，重新评估 train/val/test 三个 split，再写出 ensemble 的 `metrics.json`、`predictions_test.npz` 和预测曲线图。

## 命令

```bash
PYTHONPATH=src python scripts/run_experiment.py --config configs/train_improved_smoke_90.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/train_improved_smoke_365.yaml
```

上面两个 smoke 命令仍对应 TCN+Transformer 早期尝试。最终 ensemble 结果在完成 LSTM/Transformer 正式训练后生成：

```bash
PYTHONPATH=src python scripts/build_ensemble_results.py --device cuda:5 --force
PYTHONPATH=src python scripts/summarize_official_results.py
```

## 当前正式结果

`LSTM+Transformer Ensemble` 在两个预测窗口上都是当前最优：

- 90 天：test MSE `0.374844±0.021490`，test MAE `0.476894±0.012026`。
- 365 天：test MSE `0.444566±0.034877`，test MAE `0.516384±0.022741`。
