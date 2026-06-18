# ML Homework

本项目用于完成 2026 年专硕机器学习课程大作业：家庭电力消耗多变量时间序列预测。

## 当前目标

基于过去 90 天的多变量电力消耗与可选天气信息，预测未来每日总有功功率：

- 短期预测：未来 90 天。
- 长期预测：未来 365 天。

需要分别实现并比较：

- LSTM
- Transformer
- 自提出改进模型

## 作业要求

- 详细要求见 [docs/assignment-requirements.md](docs/assignment-requirements.md)。
- 项目执行骨架见 [docs/project-foundation.md](docs/project-foundation.md)。
- 完整复现说明见 [docs/reproducibility.md](docs/reproducibility.md)。
- 当前 Trellis 总任务见 [.trellis/tasks/06-17-ml-homework-delivery-orchestration/prd.md](.trellis/tasks/06-17-ml-homework-delivery-orchestration/prd.md)。

## 环境

本项目使用独立 conda 环境：

```bash
conda env create -f environment.yml
conda activate ml-homework
python scripts/smoke_env.py
PYTHONPATH=src pytest
```

`environment.yml` 默认安装 Python 3.10、PyTorch GPU 版和基础数据科学依赖；当前选择 `pytorch-cuda=11.8`，可在本机 Driver 535 上运行，并且命中已有本地 conda 包缓存。`scripts/smoke_env.py` 会输出 Python、PyTorch、CUDA 可用性和核心依赖版本。若当前 GPU 被占用，环境 smoke 仍可在 CPU fallback 下完成。

## 数据

当前数据源为 UCI `Individual household electric power consumption` 官方数据集：

```bash
PYTHONPATH=src python scripts/download_uci_power.py
PYTHONPATH=src python scripts/prepare_data.py --config configs/data_uci.yaml
```

流水线会生成日级聚合数据、90/365 天预测窗口、scaler 和数据摘要。默认不提交 `data/` 与 `results/` 下的大文件或生成产物。

## 训练 Smoke

共享训练框架使用本地 YAML 配置和 PyTorch：

```bash
PYTHONPATH=src python scripts/run_experiment.py --config configs/train_smoke.yaml
```

该命令会用少量 UCI `output90` 窗口跑通 train/eval/save/plot，产物写入 `results/runs/smoke_linear_output90/seed_42/`。

LSTM baseline smoke：

```bash
PYTHONPATH=src python scripts/run_experiment.py --config configs/train_lstm_smoke_90.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/train_lstm_smoke_365.yaml
```

Transformer baseline smoke：

```bash
PYTHONPATH=src python scripts/run_experiment.py --config configs/train_transformer_smoke_90.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/train_transformer_smoke_365.yaml
```

改进模型 smoke：

```bash
PYTHONPATH=src python scripts/run_experiment.py --config configs/train_improved_smoke_90.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/train_improved_smoke_365.yaml
```

## 正式实验

正式实验覆盖 LSTM、Transformer、TCN+Transformer 尝试，以及最终 LSTM+Transformer Ensemble 改进模型：

```bash
PYTHONPATH=src python scripts/run_official_experiments.py --epochs 8 --batch-size 32 --device cpu
PYTHONPATH=src python scripts/build_ensemble_results.py --device cuda:5 --force
PYTHONPATH=src python scripts/run_naive_baselines.py
PYTHONPATH=src python scripts/run_ensemble_weight_sweep.py --device cpu
PYTHONPATH=src python scripts/summarize_official_results.py
python scripts/build_design_report_pdf.py
```

完整 run 目录默认写入 `results/official_runs/` 并保持忽略；报告可引用的汇总表、manifest、naive baseline、ensemble 权重消融和对比图写入 `results/official_summary/`。最终 improved model 是固定 `0.5/0.5` 权重的 LSTM+Transformer Ensemble；`run_naive_baselines.py` 提供传统 sanity baseline，`run_ensemble_weight_sweep.py` 用验证集调权做 ablation，证明该选择不是测试集调参。

当前最终测试集结果：

| Model | Horizon | Test MSE mean±std | Test MAE mean±std |
| --- | ---: | ---: | ---: |
| LSTM+TF Ensemble | 90 | 0.374844±0.021490 | 0.476894±0.012026 |
| LSTM+TF Ensemble | 365 | 0.444566±0.034877 | 0.516384±0.022741 |

## GitHub

- 仓库链接：https://github.com/ziangbuchu/ML_homework

## 最终交付

- 完整可运行代码。
- GitHub 仓库链接。
- 包含截图和参考文献的 PDF 报告。
- 三种方法在 MSE、MAE 上的 5 轮 mean/std 对比。

报告导出优先使用 HTML/CSS 设计稿和本机 Chrome/Chromium，生成蓝色数据档案风格的 A4 PDF：

```bash
python scripts/build_design_report_pdf.py
```

该命令读取 `reports/ML_homework_report.html`，输出 `reports/ML_homework_report.pdf`，并会自动剔除 Chrome 打印偶发产生的空白页。LaTeX/Tectonic 版本仍保留为 fallback：

```bash
bash scripts/build_latex_report.sh
```

如果本机没有 `tectonic`，可按 [docs/reproducibility.md](docs/reproducibility.md) 的“报告工具”小节安装用户态二进制。旧版 Python PDF 构建脚本仅保留为最后 fallback：

```bash
PYTHONPATH=src python scripts/build_report_pdf.py --input reports/ML_homework_report.md --output reports/ML_homework_report.pdf
```

提交前自检：

```bash
PYTHONPATH=src python scripts/verify_submission.py
```
