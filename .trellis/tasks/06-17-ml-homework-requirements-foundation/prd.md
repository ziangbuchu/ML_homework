# 记录机器学习大作业要求与项目地基

## Goal

为机器学习课程大作业建立项目基础：从课程 PDF 中提取硬性要求，固化到项目文档和 Trellis task 中，明确后续代码、实验、截图、报告和 GitHub 提交的交付边界。

## What I Already Know

- 本项目用于完成 2026 年专硕机器学习课程项目。
- 提交截止时间：2026 年 7 月 15 日中午 12:00 之前。
- 提交入口：`https://docs.qq.com/form/page/DT3pqV3pNcGV6TG1z`。
- PDF 报告必须包含四部分：问题介绍、模型、结果与分析、讨论。
- 报告中必须附上完整可运行代码的 GitHub 链接。
- 结果需要以截图形式贴在报告中，并绘制 `power` 预测值与 `Ground Truth` 曲线对比图。
- 核心任务是家庭电力消耗多变量时间序列预测，目标为未来每日总有功功率。
- 数据来源指向 UCI `Individual household electric power consumption` 数据集，可结合法国月度气象数据。
- 模型任务分三部分，分值各占三分之一：
  - LSTM 模型预测。
  - Transformer 模型预测。
  - 自己提出的改进模型预测，新颖性优先，性能次之。
- 每类方法都需要分别训练短期和长期预测模型：
  - 过去 90 天预测未来 90 天。
  - 过去 90 天预测未来 365 天。
- 短期和长期模型需要分别训练，长期预测模型参数不能复用短期预测模型参数。
- 评价指标：MSE 和 MAE。
- 至少进行五轮实验，报告均值和标准差 std。

## Source Material

- 用户提供 PDF：`/data1/lf/.codex/attachments/3b1f0ea3-5c5d-40c8-b5d5-4ddb3116b7f5/2026-专硕机器学习课程考核.pdf`
- PDF 提取命令：`pdftotext -layout <pdf> -`
- 项目内整理文档：
  - `docs/assignment-requirements.md`
  - `docs/project-foundation.md`

## Requirements

### Data Requirements

- 使用课程提供的 `train.csv` 和测试集文件。PDF 文中写作 `tes.csv`，后续拿到数据后需要确认实际文件名是否为 `test.csv`。
- 原始电力数据分钟级，需要自行处理为日级样本。
- 数据可能存在缺失值，属于真实数据正常情况；处理逻辑需要显式、可复现。
- 日级聚合规则：
  - `global_active_power`、`global_reactive_power`、`sub_metering_1`、`sub_metering_2`：按天求和。
  - `voltage`、`global_intensity`：按天取平均。
  - `RR`、`NBJRR1`、`NBJRR5`、`NBJRR10`、`NBJBROU`：取当天任意一个数据。
- 可计算额外变量：
  - `sub_metering_remainder = (global_active_power * 1000 / 60) - (sub_metering_1 + sub_metering_2 + sub_metering_3)`

### Modeling Requirements

- 输入窗口：过去 90 天。
- 输出窗口：
  - 短期：未来 90 天。
  - 长期：未来 365 天。
- 必做模型：
  - LSTM baseline。
  - Transformer baseline。
  - 改进模型，结构不限，但要能说明改进原理和新颖性。
- 每种模型都需要支持两个输出长度，并确保短期/长期训练互相独立。

### Experiment Requirements

- 使用 MSE 和 MAE 两种指标。
- 每个方法至少运行五轮实验。
- 报告结果需包含 mean 和 std。
- 需要对三种方法进行比较。
- 需要输出预测曲线与 Ground Truth 曲线对比图。
- 结果截图需要最终贴入 PDF 报告。

### Report Requirements

- PDF 报告结构固定为：
  1. 问题介绍
  2. 模型，可以包含少量伪代码
  3. 结果与分析
  4. 讨论
- 报告必须包含 GitHub 链接。
- 报告必须注明参考文献。
- 如使用 ChatGPT、DeepSeek 等工具辅助报告撰写，需要注明；这些工具仅限报告撰写辅助，必要参考文献仍不可缺少。
- 若与他人组队，最多 2 人，并在报告中列明作者贡献与各自研究领域。

## Acceptance Criteria

- [x] Trellis task 已创建并包含课程要求 PRD。
- [x] 项目文档中有一份可引用的作业要求清单。
- [x] 项目文档中有一份后续目录、实验矩阵、报告产物和风险清单。
- [x] JSONL 上下文文件不再只有 seed row，并能通过 `task.py validate`。
- [x] 后续开发者能不重新打开 PDF，也能知道必须交付什么。

## Out of Scope

- 本 task 不实现模型代码。
- 本 task 不下载数据集。
- 本 task 不运行训练实验。
- 本 task 不生成最终 PDF 报告。
- 本 task 不创建 GitHub 远程仓库或 push。

## Open Questions

- 改进模型采用什么方向后续再定，候选可包括 CNN-Transformer、TCN-Transformer、LSTM-attention 或轻量残差混合模型。
- GitHub 仓库 URL 尚未确定，报告模板中先保留占位。
- 组队信息尚未确定；若单人完成，报告中仍需明确作者贡献。

## Technical Notes

- 当前项目是新目录，除 Trellis 初始化文件外暂无代码。
- 本阶段以中文记录要求，后续代码命名、README 和报告可按课程提交习惯继续使用中文说明加英文技术标识符。
