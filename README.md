# Student Health Risk Prediction

Kaggle Playground Series: Student Health Risk Prediction. Predict students' health risks based on various features.

## Overview

This repository implements a machine learning pipeline for predicting student health risks. The goal is to develop a model that can predict health risk based on various student demographics and health-related features.

## Features

The model uses various features including:
- Demographics (age, gender, etc.)
- Health metrics (BMI, blood pressure, etc.)
- Lifestyle factors (sleep, physical activity, diet quality)
- Psychological factors (stress levels, mental health)

## Architecture

- **Package**: `src/student_health/` — core ML logic, data processing, feature engineering
- **CLI**: `scripts/train.py` — pipeline orchestration
- **Data**: `data/raw/` — competition train/test CSVs
- **Experiments**: `experiments/` with `train_config.yaml` for MLflow tracking
- **Submissions**: `submissions/` for final Kaggle submissions

## Commands

```
make install        # Install dependencies and set up Kaggle auth
make download       # Fetch competition data to data/raw/
make sim-download   # Fetch sample train/test CSV files (for development)
make test           # Run tests
make test ARGS="-k name -x"  # Focused test
make lint           # Code linting
make format         # Code formatting (check)
make format-fix     # Apply ruff formatting
make list           # List dataset statistics in stats.json
```

## Key gotchas

- `from __future__ import annotations` used in all source files.
- Ruff: line-length 100, target-version = "py311", rules E/F/I.
- Python 3.12 (`.python-version`), `requires-python = ">=3.11"`.
- Kaggle auth: `.kaggle/access_token` (chmod 600) or env var `KAGGLE_API_TOKEN`. Also supports legacy `~/.kaggle/kaggle.json`.
- You **must join** the competition on Kaggle before `make download` (otherwise 403).
- No CI / GitHub Actions configured.

## Gitignored

`data/raw/*.zip`, `data/raw/*.csv`, `submissions/`, `.kaggle/*` (except `*.example`), `.ai/*`, `.venv/`, caches.

## References

- **`Iteration.md`** — experiment log, suggested directions, workflow reference. Read before starting new experiments.
- **`scripts/`** — core pipeline scripts (train, predict, evaluate).

See the kcom-hull-prediction or kcom-pokemon competitions for full details.
