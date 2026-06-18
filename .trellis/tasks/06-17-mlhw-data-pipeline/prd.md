# 实现数据获取清洗与窗口构造流水线

## Goal

把 UCI 官方分钟级家庭电力数据转换为可复现实验样本，生成 90 天输入窗口与 90/365 天输出窗口。

## Requirements

- 支持下载并读取 UCI 官方 `household_power_consumption.txt`。
- 不伪造课程文档里的 `train.csv` / `test.csv`；当前事实源是 UCI 单个分号分隔文本文件。
- 解析日期时间列，将分钟级记录按日聚合。
- 日级聚合遵循课程规则：
  - `global_active_power`、`global_reactive_power`、`sub_metering_1`、`sub_metering_2` 按天求和。
  - `voltage`、`global_intensity` 按天取平均。
  - 天气字段如使用，则按课程说明取当天可用值。
- 显式处理缺失值，并在日志/文档中记录策略。
- 构造滑动窗口：输入 90 天，输出 90 天或 365 天。
- 先生成完整滑动窗口，再按窗口顺序做 train/val/test = 70/15/15 切分。
- 每个输出长度单独拟合 scaler；scaler 只使用该任务训练窗口覆盖到的日级数据，保存 scaler 与窗口配置。
- 输出可被训练框架稳定消费的数据产物或 Dataset 接口。

## Acceptance Criteria

- [x] 给定原始 UCI 文件，能生成短期和长期任务所需样本。
- [x] 数据处理配置、缺失值策略、聚合规则可追踪。
- [x] 窗口切分按时间顺序执行；相邻窗口目标日期重叠是已接受的滑动窗口边界，需要在报告中说明。
- [x] 有最小测试或 smoke 覆盖聚合、窗口长度和 scaler 拟合边界。
- [x] 数据统计摘要保存到 `results/` 或 `docs/`，便于报告引用。

## Dependencies

- 依赖 `06-17-mlhw-env-scaffold`。

## Decision Points

- 是否融合天气数据：已决定不加入，先保持电力变量主线可复现。
- 缺失值策略：已决定分钟级 time interpolation，然后前向/后向填充，再做日聚合。
- 测试集格式：已决定使用 UCI 官方单文件，按窗口顺序切 train/val/test。
- 长期 365 天切分：已决定使用窗口级切分；严格日级非重叠切分会导致 val/test 长期样本为 0。

## Out of Scope

- 不实现具体神经网络模型。
- 不运行五轮正式实验。

## Technical Notes

- 需求来源：`docs/assignment-requirements.md` 的数据处理要求。
- 数据目录应默认不提交大文件。
- 官方数据源：`https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption`。
- 下载命令：`PYTHONPATH=src python scripts/download_uci_power.py`。
- 准备命令：`PYTHONPATH=src python scripts/prepare_data.py --config configs/data_uci.yaml`。
- 完整 prepare 验证结果：
  - 原始分钟数：2,075,259；缺失分钟行：25,979；填充后关键列缺失值为 0。
  - 日级行数：1,442；日期范围：2006-12-16 至 2010-11-26。
  - `output=90`：train 884、val 189、test 190 个窗口。
  - `output=365`：train 691、val 148、test 149 个窗口。
