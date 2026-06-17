# 课程 PDF 要求提取摘要

来源 PDF：

`/data1/lf/.codex/attachments/3b1f0ea3-5c5d-40c8-b5d5-4ddb3116b7f5/2026-专硕机器学习课程考核.pdf`

提取方式：

```bash
pdftotext -layout /data1/lf/.codex/attachments/3b1f0ea3-5c5d-40c8-b5d5-4ddb3116b7f5/2026-专硕机器学习课程考核.pdf -
```

## 硬性要求

- 截止时间：2026 年 7 月 15 日中午 12:00 前。
- PDF 报告结构：问题介绍、模型、结果与分析、讨论。
- 报告必须给出 GitHub 链接。
- 结果必须以截图形式贴入报告。
- 需要绘制 `power` 预测值与 `Ground Truth` 曲线对比图。
- 需要注明参考文献。
- 如使用 ChatGPT、DeepSeek 等工具辅助报告撰写，需要注明。

## 任务要求

- 任务：家庭电力消耗多变量时间序列预测。
- 目标：根据过去电力消耗情况，预测未来每日总有功功率。
- 输入窗口：过去 90 天。
- 输出窗口：
  - 短期预测：未来 90 天。
  - 长期预测：未来 365 天。
- 短期和长期模型需要分别训练，参数不能互用。

## 模型要求

- LSTM。
- Transformer。
- 自提出改进模型。
- 三部分各占总分三分之一。
- 改进模型以原理新颖程度为首要评价标准，性能为次要评价标准。

## 评价要求

- 指标：MSE 和 MAE。
- 每个方法至少五轮实验。
- 结果需要报告平均值和标准差 std。
- 需要比较三种方法。

## 数据处理要求

- 原始数据分钟级，需要自行处理。
- 日级聚合：
  - `global_active_power`、`global_reactive_power`、`sub_metering_1`、`sub_metering_2`：按天求和。
  - `voltage`、`global_intensity`：按天取平均。
  - `RR`、`NBJRR1`、`NBJRR5`、`NBJRR10`、`NBJBROU`：取当天任意一个数据。
- 可计算 `sub_metering_remainder`：

```text
(global_active_power * 1000 / 60) - (sub_metering_1 + sub_metering_2 + sub_metering_3)
```

## 需要后续确认

- 课程实际发放的数据文件名，尤其 PDF 中的 `tes.csv` 是否为 `test.csv`。
- GitHub 仓库链接。
- 改进模型方案。
- 单人或组队，以及贡献说明。
