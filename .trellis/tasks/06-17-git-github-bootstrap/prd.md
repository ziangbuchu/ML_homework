# git 管理与创建 GitHub 仓库

## Goal

将 `/data1/lf/work/ML_homework` 纳入 Git 管理，创建 GitHub 云端仓库，并推送当前项目地基，方便最终 PDF 报告引用 GitHub 链接。

## Current State

- 当前目录不是 Git 仓库：`git rev-parse --is-inside-work-tree` 退出码为 128。
- Git 可用：`/usr/bin/git`。
- GitHub CLI 可用：`/data1/lf/.local/bin/gh`。
- `gh auth status` 显示已登录 `github.com`，账号为 `ziangbuchu`，具备 `repo` 和 `workflow` scope。
- 精确查询 `ziangbuchu/ML_homework` 返回仓库不存在；用户确认后已创建 private 仓库。
- GitHub 仓库：`https://github.com/ziangbuchu/ML_homework`
- 当前项目大小约 1.1 MB，主要为 Trellis 初始化文件、作业要求文档和 README。

## Recommended Plan

1. 添加根目录 `.gitignore`，忽略后续机器学习项目的本地数据、模型权重、缓存和报告构建中间产物。
2. 运行 `git init -b main` 初始化本地仓库。
3. 检查 `git status --short`，确认不会纳入本地 runtime：
   - `.trellis/.developer`
   - `.trellis/.runtime/`
   - Python `__pycache__/`
4. 显式 stage 当前项目地基文件：
   - `AGENTS.md`
   - `README.md`
   - `.agents/`
   - `.codex/`
   - `.trellis/` 中可版本化文件
   - `docs/`
5. 运行基础校验：
   - `python3 ./.trellis/scripts/task.py validate 06-17-git-github-bootstrap`
   - `python3 ./.trellis/scripts/task.py validate 06-17-ml-homework-requirements-foundation`
6. 创建初始提交：
   - `chore(project): 初始化课程作业项目`
7. 使用 GitHub CLI 创建远端仓库：
   - owner: `ziangbuchu`
   - repo: `ML_homework`
   - visibility: private
   - remote: `origin`
8. 推送 `main`：
   - `git push -u origin main`

## Proposed Root `.gitignore`

```gitignore
# Local data and generated artifacts
data/
results/
checkpoints/
models/
outputs/
logs/
wandb/

# Report build outputs
reports/**/*.aux
reports/**/*.bbl
reports/**/*.bcf
reports/**/*.blg
reports/**/*.fdb_latexmk
reports/**/*.fls
reports/**/*.log
reports/**/*.out
reports/**/*.run.xml
reports/**/*.synctex.gz
reports/**/*.toc

# Python
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/

# Environments and secrets
.env
.env.*
!.env.example
.venv/
venv/

# OS/editor
.DS_Store
Thumbs.db
```

## Acceptance Criteria

- [x] 本地仓库已初始化为 `main` 分支。
- [x] 根目录 `.gitignore` 已覆盖数据、结果、checkpoint、缓存和密钥。
- [x] 初始提交只包含项目地基与 Trellis 可版本化文件，不包含本地 runtime、数据或密钥。
- [x] GitHub 仓库 `ziangbuchu/ML_homework` 已创建。
- [x] `origin` 已指向 GitHub 仓库。
- [x] `main` 已推送到远端。
- [x] 最终输出 GitHub 链接，可写入 PDF 报告。

## Explicit Confirmation Needed

执行本计划前需要用户确认：

- 是否使用仓库名 `ML_homework`。
- 是否创建为 private 仓库。
- 是否允许执行初始 commit 和首次 push。

## Out of Scope

- 不创建 PR。
- 不修改用户级 Git 或 GitHub 配置。
- 不上传课程原始数据、大模型权重、训练产物或密钥。
- 不改写 Git 历史。
