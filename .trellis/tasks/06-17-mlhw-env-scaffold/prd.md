# 搭建独立 conda 环境与项目骨架

## Goal

建立可复现的本地运行环境和项目代码骨架，为数据处理、模型训练、实验汇总和报告产物提供统一入口。

## Requirements

- 新建独立 conda 环境，推荐环境名先定为 `ml-homework`。
- 明确 Python 版本、PyTorch 安装方式和核心依赖，并写入可复现文件，例如 `environment.yml`。
- 补齐必要目录：`data/`、`scripts/`、`src/`、`results/`、`reports/`、`configs/`、`tests/`。
- 保证原始大数据、checkpoint、缓存不被提交。
- 提供最小 smoke 命令，证明环境能 import 核心依赖并运行项目入口。
- README 或项目文档中记录环境创建、激活和 smoke 命令。

## Recommended Default

- 先使用新的 conda 环境，不复用已有研究环境。
- 若本机 GPU/CUDA 适配清楚，安装 GPU 版 PyTorch；否则先建 CPU 可运行版本，后续训练 task 再升级。
- 依赖优先保持简单：`python`、`pytorch`、`numpy`、`pandas`、`scikit-learn`、`matplotlib`、`pyyaml`、`tqdm`、`pytest`。

## Acceptance Criteria

- [x] `conda env create -f environment.yml` 或等价命令可复现环境。
- [x] 项目目录骨架存在，并有合理 `.gitignore` 保护大文件。
- [x] 最小 smoke 能输出 Python、PyTorch、CUDA 可用性和关键依赖版本。
- [x] 后续子任务能直接基于该环境运行。

## Decision Points

- Python 版本：已决定使用 Python 3.10。
- 训练目标：已决定安装 PyTorch GPU 版，并保留 CPU smoke fallback。
- 环境名：已决定固定为 `ml-homework`。

## Out of Scope

- 不实现数据处理逻辑。
- 不实现模型训练。
- 不运行正式实验。

## Technical Notes

- 需要参考 `docs/project-foundation.md` 的建议目录结构。
- 进入实现前读取 `.trellis/spec/backend/index.md` 和质量规范。
- 本机 conda 路径：`/data1/lf/miniconda3/bin/conda`。
- 当前 `conda env list` 未发现 `ml-homework` 环境，适合新建。
- `nvidia-smi` 显示 Driver `535.230.02`、CUDA `12.2`；A40/L40S 多卡可用，但当前多张卡已有其他进程占用，正式训练前需要重新确认空闲 GPU。
- 首次尝试 `pytorch-cuda=12.1` 时，conda 下载 `pytorch-2.5.1-py3.10_cuda12.1...` 过慢且只有 partial cache；本机已有完整 `pytorch-2.5.1-py3.10_cuda11.8...` 缓存，因此环境文件改为 `pytorch=2.5.1` + `pytorch-cuda=11.8`。这仍是 GPU 版 PyTorch，且可由当前 driver 支持。
- 配置修正记录：曾误把 `pytorch=2.5.1` 写入 `channels`，conda 报 `UnavailableInvalidChannel`；已修正为 channel `pytorch` + dependency `pytorch=2.5.1`。
- 首次 smoke 暴露 `ImportError: ... libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`，根因是 defaults 解出 `mkl 2025.0.0` 与当前 PyTorch 2.5.1 组合不兼容；已在 `environment.yml` pin `mkl=2023.1.0`，并同步降级当前 env 的 `intel-openmp/mkl`。
- 验证通过：
  - `/data1/lf/miniconda3/envs/ml-homework/bin/python scripts/smoke_env.py` 输出 `torch=2.5.1`、`torch_cuda_available=True`、`torch_cuda_version=11.8`、`torch_device_count=8`。
  - `PYTHONPATH=src /data1/lf/miniconda3/envs/ml-homework/bin/pytest` 收集 1 个测试，`1 passed in 0.26s`。
