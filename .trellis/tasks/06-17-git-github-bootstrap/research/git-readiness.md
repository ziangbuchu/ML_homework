# Git/GitHub Readiness Notes

## Local Git

- `git rev-parse --is-inside-work-tree` returned exit code 128, so the project is not yet a Git repository.
- `git` binary is available at `/usr/bin/git`.

## GitHub CLI

- `gh` binary is available at `/data1/lf/.local/bin/gh`.
- `gh auth status` reports an authenticated session for account `ziangbuchu`.
- Token scopes include `repo` and `workflow`, enough for creating a private repository and pushing.

## Remote Repository Availability

- `gh repo view ziangbuchu/ML_homework` initially could not resolve the repository.
- After user confirmation, the private repository was created at `https://github.com/ziangbuchu/ML_homework`.

## Versioning Boundary

Should track:

- Trellis project scaffolding needed for future work.
- Project README.
- Assignment requirement docs.
- Project-scoped Codex/Trellis helpers.

Should not track:

- `.trellis/.developer`.
- `.trellis/.runtime/`.
- Python `__pycache__/`.
- Future `data/`, `results/`, `checkpoints/`, `models/`, `outputs/`, `logs/`, `wandb/`.
- `.env` or other secrets.
