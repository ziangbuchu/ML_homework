# Transformer Baseline

## 方法

Transformer baseline 使用 encoder-only 结构：

1. 输入窗口形状为 `batch_size x 90 x num_features`。
2. 先用线性层把每日多变量特征投影到 `d_model`。
3. 加入固定 sinusoidal positional encoding。
4. 使用 Transformer encoder 建模 90 天序列。
5. 对 encoder 输出做 mean pooling。
6. 用线性层投影到 `output_length`，其中 `output_length` 为 90 或 365。

当前默认参数：

- `d_model: 64`
- `nhead: 4`
- `num_layers: 2`
- `dim_feedforward: 128`
- `dropout: 0.1`

短期和长期预测使用独立 YAML 配置与独立产物目录。

## Smoke 命令

```bash
PYTHONPATH=src python scripts/run_experiment.py --config configs/train_transformer_smoke_90.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/train_transformer_smoke_365.yaml
```

正式五轮实验由 official experiments task 统一调度。
