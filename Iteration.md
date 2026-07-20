# Experiment Log

Read before starting new experiments.

## Current Status
- Repository initialized with full pipeline
- Baseline kernel at `scripts/kernels/baseline.py` → LB ~0.90 (~500/800)
- TabPFN v3 kernel at `scripts/kernels/tabpfn.py` (GPU recommended)
- **Stack kernel** at `scripts/kernels/stack.py` — HGBC/CatB/XGB/LGBM + BA weight blend
  (inspired by [kospintr stacked baseline](https://www.kaggle.com/code/kospintr/health-stacked-hgbc-catb-xgb-lgbm-baseline))
- EDA notebook at `notebooks/1_EDA.ipynb`
- Baseline notebook at `notebooks/2_Baseline.ipynb`
- TabPFN notebook at `notebooks/3_TabPFN.ipynb`
- Stack notebook at `notebooks/5_Stack.ipynb` (`make notebook-stack`)
- Configs: `config/baseline.yaml`, `config/stack.yaml`
- Metric: Balanced Accuracy (mean recall per class)

## Key Findings
- **Target**: `health_condition` — 3 classes: `fit` (5.8%), `at-risk` (85.9%), `unhealthy` (8.4%)
- **Metric**: Balanced Accuracy — predicting all `at-risk` gives only 0.33
- **Features**: 7 numeric, 6 categorical, all have missing values (1-12%)
- **LightGBM baseline**: ~0.90 LB with median/mode imputation → label encoding → interaction features → stratified 5-fold CV
- **TabPFN**: Foundation model, no preprocessing needed, stratified subsample to 15K rows, GPU accelerated

## Score Ceiling (from estimation analysis)
- **Label-noise ceiling**: 0.9674 (irreducible ~1% label flips → ~0.033 BA loss)
- **Naive missingness floor**: 0.9411 (missing rule features, mostly `stress_level`)
- **Realistic ceiling (public LB top)**: ~0.951
- **Our baseline OOF**: 0.907
- **Truly exploitable gap**: ~0.010 (after subtracting 0.033 irreducible noise)
- **Key insight**: `stress_level` is independent noise → when missing (12% of rows), can't distinguish `fit` from `unhealthy`
- See `notebooks/4_ScoreCeiling.ipynb` for full decomposition

## Next Steps
1. [ ] Review score ceiling notebook: `make notebook-score-ceiling`
2. [ ] Run TabPFN locally: `uv run python scripts/kernels/tabpfn.py` (or `make kernel-tabpfn`)
3. [ ] Submit: `make submit`
4. [ ] Close the ~0.010 exploitable gap via better imputation + threshold tuning
5. [ ] Ensemble TabPFN + LightGBM (complementary inductive biases)
6. [ ] Hyperparameter tuning with Optuna for LightGBM
7. [x] Add XGBoost + CatBoost (+ HGBC) stacked ensemble with BA weight blend
8. [ ] Experiment with SMOTE, threshold tuning
9. [ ] Try larger TabPFN subsample with `ignore_pretraining_limits=True`
10. [ ] Run stack locally / on Kaggle and record OOF BA + LB
11. [ ] Blend stack OOF with TabPFN probabilities

## Make Targets
| Command | What it does |
|---------|-------------|
| `make notebooks` | Build all .ipynb notebooks |
| `make notebook-eda` | Build EDA notebook |
| `make notebook-baseline` | Build LightGBM baseline notebook |
| `make notebook-tabpfn` | Build TabPFN notebook |
| `make notebook-score-ceiling` | Build score ceiling estimation notebook |
| `make notebook-stack` | Build stacked HGBC/CatB/XGB/LGBM notebook |
| `make kernel-baseline` | Run LightGBM baseline locally |
| `make kernel-tabpfn` | Run TabPFN locally (needs GPU) |
| `make kernel-stack` | Instructions for stack kernel / local train |

## Questions to Answer
- What is the optimal class weighting scheme?
- Can SMOTE / ADASYN improve minority recall?
- Do XGBoost or CatBoost add ensemble diversity?
- What is the best threshold for each class to maximize balanced accuracy?
- How does feature importance change with different imputation strategies?
