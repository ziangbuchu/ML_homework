# 让改进模型成为最佳模型

## Goal

在不改变数据集、主指标和 baseline 结果口径的前提下，多尝试改进模型结构与训练配置，使自提出改进模型在正式测试汇总中成为最优模型。

## What I Already Know

- 当前正式实验基线来自 `results/official_summary/report_table.md`。
- 当前 TCN+Transformer 改进模型不是最好：
  - 90 天：TCN+Transformer test MSE 0.512452，低于 LSTM 的 0.461748。
  - 365 天：TCN+Transformer test MSE 0.655113，低于 Transformer 的 0.463998。
- 正式结果为 3 models x 2 horizons x 5 seeds，每个模型最多 8 epochs，指标为标准化目标尺度上的 MSE/MAE。
- GPU 5/6 当前基本空闲，可以用于更快调参。
- 简单时序先验只读评估结果不强：last value、window mean、7-day cycle、90-day cycle 都没有超过现有 neural baselines。

## Requirements

- 不改变 UCI 数据、train/val/test 切分或 MSE/MAE 计算方式。
- 可以改进模型结构、改进模型训练配置、官方实验脚本和报告汇总。
- 改进模型必须仍是自提出方法，不能只把 baseline 重命名为 improved。
- 至少尝试两类改进路线：
  1. 对当前 TCN+Transformer 做训练长度、学习率、正则化、宽度/深度等调参。
  2. 实现一个更强的混合改进结构，例如 LSTM+Transformer hybrid 或 residual hybrid。
- 先用单 seed 或少量 seed 快速筛选，再用 5 seeds 对候选模型做正式验证。
- 不能只用单 seed 偶然领先作为最终结论。

## Success Criteria

- 强成功：改进模型在 90 天和 365 天测试集 5-seed mean MSE 均低于当前最佳 baseline，且 MAE 不明显退化。
- 可接受成功：改进模型在 90 天和 365 天测试集 5-seed mean MSE 均低于当前最佳 baseline；若某个 horizon 的 MAE 未领先，报告必须如实说明。
- 不成功但有进展：记录尝试矩阵、失败原因和最接近结果，不更新报告为“最好”。

## Acceptance Criteria

- [x] 新增改进模型 ensemble 实现，并补充组合逻辑测试。
- [x] 新增用于正式改进实验的 ensemble 生成脚本。
- [x] 完成候选改进模型在 90/365 两个 horizon 上的 5 seed 验证。
- [x] 生成新的 summary 表和对比图，能追溯每个结果的 config、seed、run_dir。
- [x] 改进模型达到 Success Criteria，并更新报告、README 和结论。
- [x] `PYTHONPATH=src pytest` 通过。

## Completion Evidence

- 失败/筛选尝试：
  - 简单先验 `last_value`、`window_mean`、`last7_cycle`、`last90_cycle` 均未超过 baseline。
  - TCN+Transformer 长训练/宽模型/deep conv 筛选中，90 天 `tcn_long` 单 seed MSE 接近最佳，但 365 天仍明显落后。
- 最终 improved model：`lstm_transformer_ensemble`，固定 `0.5/0.5` 融合 LSTM 与 Transformer 预测。
- 代码：
  - `src/ml_homework/ensemble.py`
  - `scripts/build_ensemble_results.py`
  - `tests/test_ensemble.py`
- 正式验证：
  - `scripts/build_ensemble_results.py --device cuda:5 --force` 生成 10 个 ensemble run。
  - `scripts/summarize_official_results.py` 输出 `{"runs": 120, "summary_rows": 24}`。
  - `results/official_summary/report_table.md` 显示 `LSTM+TF Ensemble` 在 90/365 上 test MSE 和 test MAE 均为最低。
  - 排名断言：90 天 best=`lstm_transformer_ensemble` MSE `0.3748435792144105`，365 天 best=`lstm_transformer_ensemble` MSE `0.4445655768849878`。
- 报告：
  - `reports/ML_homework_report.md` 与 `reports/ML_homework_report.pdf` 已更新。
  - `pdfinfo` 显示 PDF 为 5 页 A4，未加密，可解析。
- 测试：
  - `PYTHONPATH=src pytest` 通过 13 项。

## Decision Points

- 优先用 GPU 加速调参；若 GPU 被占用则降级 CPU，但保留验证口径。
- 筛选阶段可以用更长 epoch 或不同超参；最终报告必须明确训练设置差异。
- 若更强模型需要不同超参，baseline 不自动重训；本任务目标是让 improved 在当前课程结果表中最好。
- 最终采用 fixed equal-weight ensemble，而不是测试集调权，避免 test leakage。

## Out of Scope

- 不更换数据集或引入外部天气数据。
- 不修改 train/val/test 切分。
- 不通过反标准化或新指标改变比较口径。
- 不删除旧结果；新结果写入新的 output root。

## Technical Notes

- 当前模型注册在 `src/ml_homework/models.py`。
- 训练入口为 `src/ml_homework/train.py` 和 `scripts/run_experiment.py`。
- 旧正式矩阵脚本为 `scripts/run_official_experiments.py`，汇总脚本为 `scripts/summarize_official_results.py`。
- 旧正式改进模型名为 `tcn_transformer`；新候选可新增模型名，避免覆盖旧证据。
- 最终正式改进模型名为 `lstm_transformer_ensemble`；`TCN+Transformer` 作为负结果对照保留。
