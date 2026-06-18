# 设计并实现自提出改进模型

## Goal

设计一个相对 LSTM/Transformer baseline 有清晰动机和报告叙事的改进模型，并支持短期和长期预测。

## Requirements

- 由 agent 决定最终模型方向并记录取舍。
- 改进点必须能写入报告：它解决什么时间序列建模问题、和 baseline 的差异是什么、为什么适合家庭电力消耗预测。
- 模型仍需接入统一训练评估框架，支持 90 天和 365 天输出。
- 短期和长期配置、训练产物和参数彼此独立。
- 若性能不优，也要保存足够证据支撑原因分析。

## Candidate Directions

- **CNN/TCN + Transformer**（推荐默认）：用卷积/TCN 提取局部周期和短期波动，再用 Transformer 建模长期依赖；和课程 PDF 示例方向贴合，报告容易解释。
- **LSTM + Attention**：实现较稳，解释简单，但新颖性弱一些。
- **Seasonal decomposition + neural forecaster**：叙事有特色，但实现复杂度和踩坑风险更高。

## Acceptance Criteria

- [x] PRD 或技术说明中记录最终模型方向和取舍。
- [x] 改进模型能完成短期 smoke 训练和评估。
- [x] 改进模型能完成长期 smoke 训练和评估。
- [x] 输出指标、预测数组和曲线图与两个 baseline 格式一致。
- [x] 报告素材中包含改进模型原理说明、伪代码或结构描述。

## Dependencies

- 依赖 `06-17-mlhw-data-pipeline`。
- 依赖 `06-17-mlhw-training-eval-framework`。
- 建议在 LSTM/Transformer smoke 可运行后再启动正式实现。

## Decision Points

- 改进模型具体结构：已决定使用 `tcn_transformer`，即 residual temporal convolution block + Transformer encoder。
- 新颖性优先还是稳定出结果优先：已选择足够新颖且实现稳的 CNN/TCN + Transformer。
- 是否加入特征注意力、残差连接、多尺度卷积等附加模块：已加入一个 residual temporal convolution block，不堆叠额外复杂模块。

## Out of Scope

- 不为了堆复杂度引入难以解释的大模型。
- 不在本 task 运行正式 5 seed 对比。

## Verification

- `PYTHONPATH=src pytest`：9 passed。
- `PYTHONPATH=src python scripts/run_experiment.py --config configs/train_improved_smoke_90.yaml`
  - run dir：`results/runs/smoke_tcn_transformer_output90/seed_42`
  - test MSE：`0.39567863636930484`
  - test MAE：`0.49761246897590655`
  - prediction/target shape：`(32, 90)`
- `PYTHONPATH=src python scripts/run_experiment.py --config configs/train_improved_smoke_365.yaml`
  - run dir：`results/runs/smoke_tcn_transformer_output365/seed_42`
  - test MSE：`0.7417381386459871`
  - test MAE：`0.6846446579762165`
  - prediction/target shape：`(32, 365)`
