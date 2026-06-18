# 项目地基

本项目目标是完成机器学习课程大作业：家庭电力消耗多变量时间序列预测，并交付完整可运行代码和 PDF 报告。

## 交付物

- 可运行代码仓库。
- GitHub 仓库：`https://github.com/ziangbuchu/ML_homework`
- 实验配置、训练脚本、评估脚本和绘图脚本。
- 三类模型的短期与长期预测结果。
- MSE、MAE 的 5 轮 mean/std 汇总表。
- `power` 预测曲线与 `Ground Truth` 曲线截图。
- PDF 报告，包含 GitHub 链接和参考文献。

## 建议目录结构

```text
ML_homework/
  data/                  # 本地数据，不提交大文件
  docs/                  # 作业要求、计划、报告素材说明
  notebooks/             # 可选探索性分析
  reports/               # PDF 报告源文件或最终导出文件
  results/               # 实验指标、预测曲线、截图
  scripts/               # 数据处理、训练、评估、绘图入口
  src/                   # 可复用 Python 包代码
  .trellis/              # Trellis 任务与规范
```

如果后续创建 GitHub 仓库，应避免提交原始大数据、模型 checkpoint 和临时缓存；提交代码、配置、小型结果表、报告和必要截图。

## 实验矩阵

每个模型都需要覆盖两种预测长度。

| 方法 | 输入窗口 | 输出窗口 | 独立训练 | 轮数 | 指标 |
| --- | --- | --- | --- | --- | --- |
| LSTM | 90 天 | 90 天 | 是 | >= 5 | MSE, MAE |
| LSTM | 90 天 | 365 天 | 是 | >= 5 | MSE, MAE |
| Transformer | 90 天 | 90 天 | 是 | >= 5 | MSE, MAE |
| Transformer | 90 天 | 365 天 | 是 | >= 5 | MSE, MAE |
| 改进模型 | 90 天 | 90 天 | 是 | >= 5 | MSE, MAE |
| 改进模型 | 90 天 | 365 天 | 是 | >= 5 | MSE, MAE |

## 数据流水线边界

后续实现时建议固定以下流水线：

1. 下载或读取 UCI 官方 `household_power_consumption.txt`。
2. 解析 `Date` + `Time`，处理分钟级原始记录。
3. 按天聚合电力变量和天气变量。
4. 显式处理缺失值，并记录策略。
5. 构造滑动窗口样本：
   - 输入长度：90 天。
   - 输出长度：90 或 365 天。
6. 先生成完整滑动窗口，再按窗口顺序切分 train / val / test，保证 365 天长期预测也有验证和测试样本。
7. 每个输出长度单独保存标准化器、数据划分和窗口参数，保证复现实验。

## 模型边界

后续实现时建议保持统一接口：

- 输入张量：`batch_size x input_length x num_features`。
- 输出张量：`batch_size x output_length`，目标为每日总有功功率。
- LSTM、Transformer、改进模型共享训练与评估循环。
- 短期和长期预测使用独立配置与独立训练产物。

## 结果与报告素材

每次正式实验至少保存：

- 运行配置。
- 随机种子。
- 训练日志。
- 测试集 MSE 和 MAE。
- 预测数组和 Ground Truth 数组。
- 曲线图。

最终报告至少需要：

- 三类方法对比表。
- 短期预测曲线对比图。
- 长期预测曲线对比图。
- 五轮实验 mean/std 表。
- 对改进模型新颖性和结果原因的讨论。

## 当前风险与待确认项

- 数据源已确认使用 UCI 官方单文件；课程 `tes.csv` / `test.csv` 拼写差异不再作为当前主线 blocker。
- GitHub 仓库已创建：`https://github.com/ziangbuchu/ML_homework`
- 改进模型已确定为固定 `0.5/0.5` 权重的 LSTM+Transformer Ensemble。
- 是否组队尚未确定；若组队，需要记录贡献。
- PDF 报告默认由 `reports/ML_homework_report.html` 通过 `scripts/build_design_report_pdf.py` 导出；LaTeX 和旧 Python 构建脚本仅作为 fallback。
