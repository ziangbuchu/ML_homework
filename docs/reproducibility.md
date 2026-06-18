# Reproducibility Guide

本文档面向复查者说明如何从零复现实验与提交产物。默认工作目录为仓库根目录。

## 1. 环境

```bash
conda env create -f environment.yml
conda activate ml-homework
python scripts/smoke_env.py
PYTHONPATH=src pytest
```

当前环境使用 Python 3.10、PyTorch 2.5.1 和 `pytorch-cuda=11.8`。如果没有可用 GPU，可以把训练命令中的 `--device` 改为 `cpu`，但运行时间会更长。

## 2. 数据

```bash
PYTHONPATH=src python scripts/download_uci_power.py
PYTHONPATH=src python scripts/prepare_data.py --config configs/data_uci.yaml
```

数据源为 UCI `Individual household electric power consumption`。处理流程会将分钟级数据补齐、插值并聚合为日级样本，再构造 `input90_output90` 和 `input90_output365` 两组窗口。

## 3. 正式实验

先运行三类训练型模型：

```bash
PYTHONPATH=src python scripts/run_official_experiments.py --epochs 8 --batch-size 32 --device cpu --force
```

再基于已训练的 LSTM 与 Transformer 权重生成最终改进模型：

```bash
PYTHONPATH=src python scripts/build_ensemble_results.py --device cuda:5 --force
```

如果本机没有 `cuda:5`，可改为 `--device cpu`。该脚本不重新训练 baseline，而是加载同一 seed 下的 LSTM 和 Transformer `model.pt`，用固定 `0.5/0.5` 权重重新评估 train/val/test。

随后运行 ensemble 权重选择消融。该步骤只使用验证集选择权重，测试集最优权重仅作为 diagnostic oracle 写入结果文件，不用于最终模型选择：

```bash
PYTHONPATH=src python scripts/run_naive_baselines.py
PYTHONPATH=src python scripts/run_ensemble_weight_sweep.py --device cpu
```

## 4. 汇总与报告

```bash
PYTHONPATH=src python scripts/summarize_official_results.py
python scripts/build_design_report_pdf.py
```

汇总产物位于 `results/official_summary/`：

- `report_table.md`：报告中使用的测试集 mean/std 表。
- `metrics_summary.csv`：train/val/test 全部汇总。
- `metrics_by_run.csv`：每个 model/horizon/seed/split 的原始指标。
- `naive_baselines.csv`、`naive_baseline_summary.md`：last-value 和 input-mean 传统 sanity baseline。
- `ensemble_weight_sweep.csv`：LSTM/Transformer ensemble 权重网格的 val/test 指标。
- `ensemble_weight_selection.csv`：每个 horizon/seed 的验证集选权重和测试集诊断对照。
- `ensemble_weight_summary.md`：报告中使用的权重消融摘要。
- `figures/comparison_h90_seed1.png`、`figures/comparison_h365_seed1.png`：报告曲线图。

### 报告工具

最终 PDF 默认使用 `reports/ML_homework_report.html` 中的 HTML/CSS 设计稿，并通过本机 `google-chrome` 或 Chromium headless 导出到 `reports/ML_homework_report.pdf`：

```bash
python scripts/build_design_report_pdf.py
```

该脚本会检查 PDF 可解析性，并在 Chrome 打印产生空白文本页时用 `pdfseparate`/`pdfunite` 自动重组非空页面。当前机器已存在 `/usr/bin/google-chrome`，不需要额外安装系统包。

LaTeX/Tectonic 版本从 `reports/ML_homework_report.tex` 编译到同一 PDF 路径，作为 fallback 保留。如果系统或 conda 环境里没有 `tectonic`，可以安装用户态静态二进制，不需要 `sudo`：

```bash
mkdir -p .tools/tectonic
curl -L https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.16.9/tectonic-0.16.9-x86_64-unknown-linux-musl.tar.gz \
  | tar -xz -C .tools/tectonic tectonic
chmod +x .tools/tectonic/tectonic
bash scripts/build_latex_report.sh
```

旧版 `scripts/build_report_pdf.py` 仍保留为无浏览器、无 LaTeX 环境时的最后 fallback，但最终提交建议使用 HTML/CSS 设计版 PDF。

## 5. 提交前自检

```bash
PYTHONPATH=src python scripts/verify_submission.py
```

该命令检查：

- PDF 是否存在、可解析且为 A4。
- PDF 是否包含空白文本页。
- 报告是否包含 GitHub 链接、UCI 链接、参考文献和工具辅助说明。
- `lstm_transformer_ensemble` 是否在 90 天和 365 天测试集 MSE/MAE 上均排名第一。
- 关键脚本、报告和结果图是否存在。

当前最终结论：

| Model | Horizon | Test MSE mean±std | Test MAE mean±std |
| --- | ---: | ---: | ---: |
| LSTM+TF Ensemble | 90 | 0.374844±0.021490 | 0.476894±0.012026 |
| LSTM+TF Ensemble | 365 | 0.444566±0.034877 | 0.516384±0.022741 |

## 6. 结果边界

- 指标在标准化目标尺度上计算。
- 数据切分为滑动窗口按样本时间顺序切分，相邻窗口之间存在重叠。
- 最终 improved model 是固定等权重 ensemble，不是测试集调权。
- `data/`、完整 run 目录和 checkpoint 默认不进入 Git；报告可引用的小型汇总产物保留在 `results/official_summary/`。
