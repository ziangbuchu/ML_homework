# 数据流水线说明

## 数据源

当前流水线使用 UCI Machine Learning Repository 的 `Individual household electric power consumption` 数据集。

- 官方页面：`https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption`
- 原始文件：`household_power_consumption.txt`
- 原始粒度：1 分钟
- 时间范围：2006-12 至 2010-11
- 字段数：9
- 缺失值：官方说明约 1.25% 行存在缺失

## 决策

- 不融合天气数据，先保持电力变量主线可复现。
- 不伪造课程文档里的 `train.csv` / `test.csv`；UCI 官方数据是单个分号分隔文本文件。
- 先从完整日级序列生成滑动窗口，再按窗口顺序切分 train / val / test。这个策略保证 365 天长期预测也有验证和测试样本。
- 每个输出长度单独拟合 scaler；scaler 只使用该任务训练窗口覆盖到的日级数据。
- 缺失值在分钟级使用 time interpolation，再做前向/后向填充，然后聚合到日级；这样避免先日求和导致缺失分钟被当作 0 而低估当天消耗。

## 聚合规则

- 日求和：`global_active_power`、`global_reactive_power`、`sub_metering_1`、`sub_metering_2`、`sub_metering_3`、`sub_metering_remainder`
- 日平均：`voltage`、`global_intensity`
- 派生变量：

```text
sub_metering_remainder =
  global_active_power * 1000 / 60
  - sub_metering_1
  - sub_metering_2
  - sub_metering_3
```

## 命令

```bash
PYTHONPATH=src python scripts/download_uci_power.py
PYTHONPATH=src python scripts/prepare_data.py --config configs/data_uci.yaml
```

输出默认写入：

- 原始数据：`data/raw/uci_power/`
- 日级数据与窗口：`data/processed/uci_power/`
- 数据摘要：`results/data/uci_power_summary.json`

## 切分说明

`configs/data_uci.yaml` 当前使用：

```yaml
split:
  train_ratio: 0.70
  val_ratio: 0.15
  policy: chronological_sliding_window_splits
```

对每个输出长度分别生成全部滑动窗口，然后按窗口顺序切分。相邻窗口会共享部分输入或预测日期，这是多步时间序列滑动窗口训练的常见设置；报告中需要说明该边界。
