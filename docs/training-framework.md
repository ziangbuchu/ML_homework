# 训练评估框架说明

## 设计决策

- 使用本地 YAML 配置，不引入 WandB 等外部服务。
- 所有模型共享同一个 Dataset、训练循环、指标、保存和绘图逻辑。
- 运行产物落在 `results/runs/<experiment>/seed_<seed>/`。
- 当前仅内置 `linear` smoke 模型；LSTM、Transformer、改进模型后续通过 `ml_homework.models.build_model` 接入。

## 命令

```bash
PYTHONPATH=src python scripts/run_experiment.py --config configs/train_smoke.yaml
```

默认 smoke 使用已准备好的 UCI `output90` 窗口，每个 split 只取少量样本，用 CPU 跑通完整链路。

## 产物

每次运行保存：

- `config.yaml`
- `metrics.json`
- `history.csv`
- `predictions_test.npz`
- `model.pt`
- `figures/prediction_*.png`

指标统一使用 `src/ml_homework/metrics.py` 中的 MSE 和 MAE。
