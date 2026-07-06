# Experiment Log

Read before starting new experiments.

## Current Status
- Repository initialized with full pipeline
- Baseline kernel at `scripts/kernels/baseline.py` → LB ~0.90 (~500/800)
- TabPFN v3 kernel at `scripts/kernels/tabpfn.py` (GPU recommended)
- EDA notebook at `notebooks/1_EDA.ipynb`
- Baseline notebook at `notebooks/2_Baseline.ipynb`
- TabPFN notebook at `notebooks/3_TabPFN.ipynb`
- Config-driven training via `config/baseline.yaml`
- Metric: Balanced Accuracy (mean recall per class)

## Key Findings
- **Target**: `health_condition` — 3 classes: `fit` (5.8%), `at-risk` (85.9%), `unhealthy` (8.4%)
- **Metric**: Balanced Accuracy — predicting all `at-risk` gives only 0.33
- **Features**: 7 numeric, 6 categorical, all have missing values (1-12%)
- **LightGBM baseline**: ~0.90 LB with median/mode imputation → label encoding → interaction features → stratified 5-fold CV
- **TabPFN**: Foundation model, no preprocessing needed, stratified subsample to 15K rows, GPU accelerated

## Next Steps
1. Run TabPFN locally: `uv run python scripts/kernels/tabpfn.py` (or `make kernel-tabpfn`)
2. Submit: `make submit`
3. Ensemble TabPFN + LightGBM (complementary inductive biases)
4. Hyperparameter tuning with Optuna for LightGBM
5. Add XGBoost + CatBoost for ensemble diversity
6. Experiment with SMOTE, threshold tuning
7. Try larger TabPFN subsample with `ignore_pretraining_limits=True`

## Make Targets
| Command | What it does |
|---------|-------------|
| `make notebooks` | Build all .ipynb notebooks |
| `make notebook-eda` | Build EDA notebook |
| `make notebook-baseline` | Build LightGBM baseline notebook |
| `make notebook-tabpfn` | Build TabPFN notebook |
| `make kernel-baseline` | Run LightGBM baseline locally |
| `make kernel-tabpfn` | Run TabPFN locally (needs GPU) |

## Questions to Answer
- What is the optimal class weighting scheme?
- Can SMOTE / ADASYN improve minority recall?
- Do XGBoost or CatBoost add ensemble diversity?
- What is the best threshold for each class to maximize balanced accuracy?
- How does feature importance change with different imputation strategies?
