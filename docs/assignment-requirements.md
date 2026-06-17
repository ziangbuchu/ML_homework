# 机器学习课程大作业要求

本文档根据用户提供的 `2026-专硕机器学习课程考核.pdf` 整理，用作后续代码、实验和报告写作的需求源。

## 提交要求

- 截止时间：2026 年 7 月 15 日中午 12:00 之前，晚于此时间视为未提交。
- 提交链接：`https://docs.qq.com/form/page/DT3pqV3pNcGV6TG1z`
- 需要提交 PDF 作业报告。
- 需要提交完整可运行代码，并在 PDF 中附上 GitHub 链接。
- 结果需以截图形式贴在报告中。
- 必须注明参考文献，否则视为抄袭。
- 允许使用 ChatGPT、DeepSeek 一类工具辅助报告撰写，但仅限撰写部分，并需注明。
- 最多 2 人组队；如组队，需要在报告中列明各作者贡献和各自所属研究领域。

## 报告结构

PDF 报告必须由以下四部分构成：

1. 问题介绍
2. 模型，可以包含少量伪代码
3. 结果与分析
4. 讨论

## 问题定义

任务是家庭电力消耗多变量时间序列预测。

目标问题：根据最近的电力消耗情况，预测接下来每一天的总有功功率。

具体任务：基于过去 90 天的数据曲线预测未来电力消耗变化曲线，包含两种输出长度：

- 短期预测：未来 90 天。
- 长期预测：未来 365 天。

注意：短期预测和长期预测需要分别训练，长期预测模型参数不能用于短期预测。

## 数据来源

典型数据集：

- UCI Machine Learning Repository: `Individual household electric power consumption`
- 链接：`https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption`
- 记录时间：2006 年 12 月至 2010 年 11 月。
- 原始粒度：每分钟。
- 内容：全屋有功功率、无功功率、电流、电压、各路子表能耗等变量。

可融合外部天气因素：

- 法国月度基础气候数据：`https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles`

课程数据集：

- PDF 指出数据主要分为 `train.csv` 和测试集文件。
- PDF 文中测试集文件写作 `tes.csv`，拿到真实数据后需要确认是否为 `test.csv`。

## 字段含义

- `global_active_power`：全局有功功率，单位 kW，表示实际消耗电能。
- `global_reactive_power`：全局无功功率，单位 kW，表示储存在电路中并来回转换的能量。
- `voltage`：电压，单位 V。
- `global_intensity`：电流强度，单位 A。
- `sub_metering_1`：厨房区域有功能量消耗，单位 Wh。
- `sub_metering_2`：洗衣房区域有功能量消耗，单位 Wh。
- `sub_metering_3`：气候控制系统有功能量消耗，单位 Wh。
- `RR`：月累计降水高度，单位为毫米的十分之一，记录值需除以 10。
- `NBJRR1`：当月日降水大于等于 1 mm 的天数。
- `NBJRR5`：当月日降水大于等于 5 mm 的天数。
- `NBJRR10`：当月日降水大于等于 10 mm 的天数。
- `NBJBROU`：当月雾出现的天数。

## 数据处理要求

原始电力数据基本时间单位是分钟，需要自行处理为建模样本。

PDF 给出的日级聚合规则：

- `global_active_power`、`global_reactive_power`、`sub_metering_1`、`sub_metering_2`：按天取总和。
- `voltage`、`global_intensity`：按天取平均。
- `RR`、`NBJRR1`、`NBJRR5`、`NBJRR10`、`NBJBROU`：取当天的任意一个数据。

可计算派生变量：

```text
sub_metering_remainder =
  (global_active_power * 1000 / 60)
  - (sub_metering_1 + sub_metering_2 + sub_metering_3)
```

缺失值说明：

- 所提供数据中可能存在缺失项。
- 真实数据有缺失是正常现象。
- PDF 说明缺失不应影响预测任务，但代码中仍需要有明确、可复现的数据处理方式。

## 模型要求

作业分为三部分，各占总分三分之一：

1. 使用 LSTM 模型进行预测。
2. 使用 Transformer 模型进行预测。
3. 使用自己提出的改进模型进行预测。

改进模型要求：

- 结构不限。
- PDF 示例方向：结合卷积层提取局部特征后接 Transformer 编码，以改进长期依赖建模能力。
- 此部分以原理的新颖程度为首要评价标准，性能为次要评价标准。
- 若方法新颖但性能不佳，只要原因分析有力，也可以获得较高分数。

## 实验与评价要求

- 指标：MSE 与 MAE。
- 每个方法至少进行五轮实验。
- 结果取平均值。
- 提供标准差 std，以评估稳定性。
- 需要比较三种方法。
- 需要绘制 `power` 预测与 `Ground Truth` 曲线对比图。
- 结果截图需要贴入 PDF 报告。

## 参考链接

PDF 中给出的参考博客：

- `https://blog.csdn.net/qq_47885795/article/details/143462299`
- `https://blog.csdn.net/weixin_39653948/article/details/105431099`
- `https://datac.blog.csdn.net/article/details/105928752?fromshare=blogdetail&sharetype=blogdetail&sharerId=105928752&sharerefer=PC&sharesource=weixin_44709585&sharefrom=from_link`

