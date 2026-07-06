# Experiment Log

Read before starting new experiments.

## Current Status
- Repository initialized with full pipeline
- Benchmark kernel created at `scripts/kernels/baseline.py`
- EDA notebook builder at `scripts/nbs/build_eda.py`
- Baseline notebook builder at `scripts/nbs/build_baseline.py`
- Config-driven training via `config/baseline.yaml`
- Model: LightGBM multiclass with `class_weight='balanced'`
- Metric: Balanced Accuracy (mean recall per class)

## Key Findings
- **Target**: `health_condition` — 3 classes: `fit` (5.8%), `at-risk` (85.9%), `unhealthy` (8.4%)
- **Metric**: Balanced Accuracy — predicting all `at-risk` gives only 0.33
- **Features**: 7 numeric, 6 categorical, all have missing values (1-12%)
- **Baseline approach**: median/mode imputation → label encoding → interaction features → stratified 5-fold LightGBM with class weights

## Next Steps
1. Run baseline locally: `uv run python scripts/kernels/baseline.py`
2. Build notebooks: `make notebooks`
3. Tune hyperparameters with Optuna
4. Add XGBoost + CatBoost for ensemble
5. Experiment with target encoding, SMOTE, threshold tuning
6. Generate submission and validate with `make submit`

## Make Targets
| Command | What it does |
|---------|-------------|
| `make notebooks` | Build EDA + baseline .ipynb notebooks |
| `make notebook-eda` | Build EDA notebook only |
| `make notebook-baseline` | Build baseline notebook only |
| `make kernel-baseline` | Run vendored Kaggle kernel locally |
| `make train` | Run config-driven training |

## Questions to Answer
- What is the optimal class weighting scheme?
- Can SMOTE / ADASYN improve minority recall?
- Do XGBoost or CatBoost add ensemble diversity?
- What is the best threshold for each class to maximize balanced accuracy?
- How does feature importance change with different imputation strategies?
