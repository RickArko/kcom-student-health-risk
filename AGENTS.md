AGENTS.md — kcom-student-health-risk

Kaggle Playground Series: Student Health Risk Prediction. Predict students' health risks based on various features.

## Commands

All Python **must** be prefixed with `uv run` (`.venv` not PATH).

| Command | What it does |
|---|---|---|
| `make install` | `uv sync --extra dev` + editable install + auth check |
| `make download` | Fetch competition data to `data/raw/` |
| `make test` | `uv run pytest tests/ -v` |
| `make test ARGS="-k name -x"` | Focused test |
| `make lint` | `ruff check src/ scripts/ tests/` (no autofix) |
| `make format` | `ruff format --check` (no rewrite) |
| `make format-fix` | Apply ruff formatting |
| `make list` | List dataset statistics in `stats.json` |
| `make train-ensemble` | LGBM+XGB+CatBoost OOF blend → `data/submissions/submission.csv` |
| `make verify-autogluon` | AutoGluon OOF bar (`uv sync --extra dev --extra verify`) |
| `make verify-tabpfn` | TabPFN subsample OOF bar (needs `TABPFN_TOKEN`) |
| `make compare` | Compare ensemble vs AutoGluon/TabPFN OOF |
| `make submit` | Submit `data/submissions/submission.csv` to Kaggle |
| `make notebook-stack` | Build stacked HGBC/CatB/XGB/LGBM notebook |
| `make kernel-stack` | Kaggle paste path for `scripts/kernels/stack.py` |

## Verification loop: `make lint && make format && make test`. No typechecker configured.

## Architecture

- **Package**: `src/student_health/` — core ML logic, data processing, feature engineering
- **CLI**: `scripts/train.py` — pipeline orchestration
- **Data**: `data/raw/` — competition train/test CSVs
- **Experiments**: `experiments/` with `train_config.yaml` for MLflow tracking
- **Submissions**: `submissions/` for final Kaggle submissions

## Key gotchas

- `from __future__ import annotations` used in all source files.
- Ruff: line-length 100, `target-version = "py311"`, rules E/F/I.
- Python 3.12 (`.python-version`), `requires-python = ">=3.11"`.
- Kaggle auth: `.kaggle/access_token` (chmod 600) or env var `KAGGLE_API_TOKEN`. Also supports legacy `~/.kaggle/kaggle.json`.
- You **must join** the competition on Kaggle before `make download` (otherwise 403).
- No CI / GitHub Actions configured.

## Gitignored

`data/raw/*.zip`, `data/raw/*.csv`, `submissions/`, `.kaggle/*` (except `*.example`), `.ai/*`, `.venv/`, caches.

See the kcom-hull-prediction or kcom-pokemon competitions for full details.
