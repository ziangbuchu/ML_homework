# LSTM Baseline

## 方法

LSTM baseline 使用 encoder-only 结构：

1. 输入窗口形状为 `batch_size x 90 x num_features`。
2. 单层 LSTM 读取 90 天多变量序列。
3. 取最后一层最后 hidden state。
4. 用线性层直接投影到 `output_length`，其中 `output_length` 为 90 或 365。

当前默认参数：

- `hidden_size: 64`
- `num_layers: 1`
- `dropout: 0.0`

短期和长期预测使用独立 YAML 配置与独立产物目录。

## Smoke 命令

```bash
PYTHONPATH=src python scripts/run_experiment.py --config configs/train_lstm_smoke_90.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/train_lstm_smoke_365.yaml
```

正式五轮实验由 official experiments task 统一调度。
