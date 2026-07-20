# Experiment Log

Read before starting new experiments.

## Current Status
- Repository initialized with full pipeline
- Baseline kernel at `scripts/kernels/baseline.py` → LB ~0.90 (~500/800), OOF ~0.907
- TabPFN v3 kernel at `scripts/kernels/tabpfn.py` (GPU recommended)
- **BA-verified GBDT ensemble** at `scripts/train_ensemble.py` / `scripts/kernels/ensemble.py`
  - LightGBM + XGBoost + CatBoost, fixed stratified folds, hill-climb blend, BA thresholds
  - Config: `config/ensemble.yaml`
- Stack kernel (HGBC/CatB/XGB/LGBM) still available at `scripts/kernels/stack.py`
- Verification baselines: `scripts/verify_autogluon.py`, `scripts/verify_tabpfn.py`, `scripts/compare_baselines.py`
- Metric: Balanced Accuracy (mean recall per class)

## Latest Ensemble Run (2026-07-19)
| Model | OOF BA |
|-------|--------|
| LightGBM | 0.9481 |
| XGBoost | 0.9494 |
| CatBoost | 0.9492 |
| **Blend + thresholds** | **0.9497** |
| AutoGluon verify (100k sample, 3-fold, 180s/fold) | 0.8790 |
| TabPFN verify (4k subsample CV, CPU) | 0.8570 |

- **Blend weights**: xgb 0.617 / catboost 0.372 / lgbm 0.011
- **Thresholds**: `[0.7, 1.0, 0.7]` for `[fit, at-risk, unhealthy]`
- **Submission**: `data/submissions/submission.csv`
- **Gate**: `make compare` → PASS (blend ≫ AutoGluon and TabPFN verify runs)
- **Hypothesis confirmed**: missing indicators + class-balanced GBDTs + OOF blend close most of the gap from 0.907 → ~0.950

## Key Findings
- **Target**: `health_condition` — 3 classes: `fit` (5.8%), `at-risk` (85.9%), `unhealthy` (8.4%)
- **Metric**: Balanced Accuracy — predicting all `at-risk` gives only 0.33
- **Features**: 7 numeric, 6 categorical, all have missing values (1-12%); missing indicators help
- **Class order**: always `fit=0, at-risk=1, unhealthy=2` (never LabelEncoder alpha order)
- **LightGBM alone** with missing indicators already reaches ~0.948–0.949 OOF
- **TabPFN / AutoGluon** verify scripts are intentionally lighter (subsample / time-limited); use them as automation bars, not as full-data ceilings

## Score Ceiling (from estimation analysis)
- **Label-noise ceiling**: 0.9674 (irreducible ~1% label flips → ~0.033 BA loss)
- **Naive missingness floor**: 0.9411 (missing rule features, mostly `stress_level`)
- **Realistic ceiling (public LB top)**: ~0.951
- **Our ensemble OOF**: 0.9497
- **Remaining gap to realistic top**: ~0.001–0.002
- **Key insight**: `stress_level` is independent noise → when missing (12% of rows), can't distinguish `fit` from `unhealthy`
- See `notebooks/4_ScoreCeiling.ipynb` for full decomposition

## Next Steps
1. [x] Build LGBM/XGB/CatBoost OOF ensemble with BA blend + thresholds
2. [x] Verify vs AutoGluon + TabPFN (`make compare`)
3. [ ] Submit ensemble: `make submit`
4. [ ] Optional: GPU TabPFN full-data OOF and blend into ensemble
5. [ ] Optional: Optuna HPO on XGB/CatBoost around current params
6. [ ] Optional: target encoding / richer FE if LB stalls below 0.950

## Make Targets
| Command | What it does |
|---------|-------------|
| `make train-ensemble` | Full LGBM+XGB+CatBlend → `data/submissions/submission.csv` |
| `make verify-autogluon` | AutoGluon OOF bar (`uv sync --extra dev --extra verify`) |
| `make verify-tabpfn` | TabPFN subsample OOF bar (needs `TABPFN_TOKEN`) |
| `make compare` | Print OOF table; fail if blend < AutoGluon |
| `make kernel-ensemble` | Run Kaggle-style ensemble kernel locally |
| `make kernel-baseline` | Run LightGBM baseline locally |
| `make kernel-tabpfn` | Run TabPFN locally (needs GPU) |
| `make kernel-stack` | Instructions for stack kernel / local train |
| `make notebooks` | Build all .ipynb notebooks |

## Questions to Answer
- Does public LB track the 0.9497 OOF closely?
- Can GPU TabPFN on larger samples add blend diversity?
- What is the best threshold schedule under nested CV (avoid OOF optimism)?
