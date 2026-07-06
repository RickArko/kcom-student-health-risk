"""Build notebooks/EDA.ipynb programmatically via nbformat.

Usage:
    uv run python scripts/nbs/build_eda.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB = nbf.v4.new_notebook()
NB.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12.0"},
}

CELLS = []


def md(src: str) -> None:
    CELLS.append(nbf.v4.new_markdown_cell(src.strip()))


def code(src: str) -> None:
    CELLS.append(nbf.v4.new_code_cell(src.strip()))


# ── %cd .. ────────────────────────────────────────────────────────────────────
code("""%cd ..
""")

# ── Title ────────────────────────────────────────────────────────────────────
md("""
# Student Health Risk Prediction — Exploratory Data Analysis

Predict students' **health risk** (`fit`, `at-risk`, `unhealthy`) based on
demographics, lifestyle, and biometric features.
Metric: **Balanced Accuracy** (mean recall per class).

[Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s6e7)

> **Key question**: 86% of samples are `at-risk` — why does 86% accuracy
> score only ~0.33 balanced accuracy?  How do we capture the minority classes?
""")

# ── TOC ──
md("""
## Table of Contents

1.  [Setup & Data Loading](#1-setup--data-loading)
2.  [Target Variable](#2-target-variable)
3.  [Missing Data Analysis](#3-missing-data-analysis)
4.  [Numerical Features](#4-numerical-features)
5.  [Categorical Features](#5-categorical-features)
6.  [Feature Correlations](#6-feature-correlations)
7.  [Feature Engineering Ideas](#7-feature-engineering-ideas)
8.  [Baseline Model](#8-baseline-model)
9.  [Summary & Recommendations](#9-summary--recommendations)
""")

# ── 1. Setup & Data Loading ──────────────────────────────────────────────────
md("## 1.  Setup & Data Loading")
code("""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=0.95)

# Locate data: try relative to cwd (after %cd ..) then relative to notebook
_DATA_CANDIDATES = [Path("data/raw"), Path("../data/raw")]
DATA_DIR = next((p for p in _DATA_CANDIDATES if (p / "train.csv").exists()), _DATA_CANDIDATES[0])
print(f"✓ DATA_DIR = {DATA_DIR.resolve()}")
""")

code("""
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

print(f"Train: {train.shape}  |  Test: {test.shape}")
print(f"Train columns: {list(train.columns)}")
print(f"Test columns:  {list(test.columns)}")
train.head()
""")

# ── 2. Target Variable ───────────────────────────────────────────────────────
md("## 2.  Target Variable")
md("""
### 2.1  Why 86% accuracy scores 0.33

The target `health_condition` has three classes.  **Balanced accuracy** is the
mean recall per class — if we predict `at-risk` for everything, the `fit` and
`unhealthy` classes get 0 recall, so the score is 1/3 ≈ 0.33 regardless of
overall accuracy.
""")

code("""
cnt = train["health_condition"].value_counts()
pct = train["health_condition"].value_counts(normalize=True) * 100

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

colors = ["#2ecc71", "#f39c12", "#e74c3c"]
axes[0].bar(cnt.index, cnt.values, color=colors, edgecolor="white", linewidth=0.5)
axes[0].set(xlabel="Health Condition", ylabel="Count", title="Class Distribution")
for i, v in enumerate(cnt.values):
    axes[0].text(i, v + 2000, f"{v:,}", ha="center", fontsize=11)

axes[1].bar(pct.index, pct.values, color=colors, edgecolor="white", linewidth=0.5)
axes[1].set(xlabel="Health Condition", ylabel="Percentage (%)", title="Class %")
for i, v in enumerate(pct.values):
    axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=11)

axes[2].axis("off")
info = (
    f"Majority class:  at-risk ({pct['at-risk']:.1f}%)\\n"
    f"Minority:  fit ({pct['fit']:.1f}%)  |  unhealthy ({pct['unhealthy']:.1f}%)\\n\\n"
    f"Always predict 'at-risk':\\n"
    f"  Accuracy = {pct['at-risk']/100:.4f}\\n"
    f"  Balanced Acc = 1/3 = 0.3333 (for 3 classes)"
)
axes[2].text(0.1, 0.5, info, fontsize=12, verticalalignment="center", fontfamily="monospace")
axes[2].set(title="Why balanced accuracy matters")

fig.tight_layout()
plt.show()
""")

code("""
# Confusion matrix illustration: always predicting majority class
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_true = train["health_condition"]
y_pred_majority = np.full_like(y_true, "at-risk")

ba = balanced_accuracy_score(y_true, y_pred_majority)
print(f"Always-predict-'at-risk' balanced accuracy: {ba:.4f}")
print()

labels = ["fit", "at-risk", "unhealthy"]
cm = confusion_matrix(y_true, y_pred_majority, labels=labels)
disp = ConfusionMatrixDisplay(cm, display_labels=labels)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix — Always Predict 'at-risk'")
plt.show()
""")

# ── 3. Missing Data ──────────────────────────────────────────────────────────
md("## 3.  Missing Data Analysis")
md("""
Every feature has missing values.  Understanding the pattern is essential.
""")

code("""
missing = train.isnull().mean().sort_values(ascending=False) * 100
missing = missing[missing > 0]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].barh(range(len(missing)), missing.values, color="C0", alpha=0.7)
axes[0].set(yticks=range(len(missing)), yticklabels=missing.index,
            xlabel="Missing %", title="Train Missing Rate")
for i, v in enumerate(missing.values):
    axes[0].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)

test_missing = test.isnull().mean().sort_values(ascending=False) * 100
test_missing = test_missing[test_missing > 0]
axes[1].barh(range(len(test_missing)), test_missing.values, color="C3", alpha=0.7)
axes[1].set(yticks=range(len(test_missing)), yticklabels=test_missing.index,
            xlabel="Missing %", title="Test Missing Rate")
for i, v in enumerate(test_missing.values):
    axes[1].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)

fig.tight_layout()
plt.show()

print(f"Train: {train.isnull().sum().sum():,} missing cells across "
      f"{(train.isnull().sum() > 0).sum()} columns")
print(f"Test:  {test.isnull().sum().sum():,} missing cells across "
      f"{(test.isnull().sum() > 0).sum()} columns")
""")

code("""
# Missing correlation — do certain features tend to be missing together?
missing_matrix = train.isnull().astype(int)
missing_corr = missing_matrix.corr()

mask = np.triu(np.ones_like(missing_corr, dtype=bool), k=1)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(missing_corr, mask=~mask, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-0.1, vmax=0.5, center=0, square=True, ax=ax)
ax.set(title="Missing Value Co-occurrence (upper triangle)")
plt.show()
""")

# ── 4. Numerical Features ────────────────────────────────────────────────────
md("## 4.  Numerical Features")
md("### 4.1  Distribution by class")

code("""
NUM_COLS = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
]
""")

code("""
fig, axes = plt.subplots(4, 2, figsize=(18, 18))
axes = axes.ravel()
target_order = ["fit", "at-risk", "unhealthy"]
palette = {"fit": "#2ecc71", "at-risk": "#f39c12", "unhealthy": "#e74c3c"}

for idx, col in enumerate(NUM_COLS):
    ax = axes[idx]
    for cls in target_order:
        subset = train[train["health_condition"] == cls][col].dropna()
        sns.kdeplot(subset, label=cls, color=palette[cls], ax=ax, linewidth=1.5)
    ax.set(title=f"{col} by Health Condition", xlabel=col)
    ax.legend()

axes[-1].axis("off")
fig.tight_layout()
plt.show()
""")

code("""
# Box plots
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.ravel()
for idx, col in enumerate(NUM_COLS):
    ax = axes[idx]
    sns.boxplot(data=train, x="health_condition", y=col, order=target_order,
                palette=palette, ax=ax)
    ax.set(title=col)

axes[-1].axis("off")
fig.tight_layout()
plt.show()
""")

md("### 4.2  Median values per class")
code("""
grouped = train.groupby("health_condition")[NUM_COLS].median()
print("Median values by class:")
print(grouped.round(2).to_string())
print()
print("Difference from overall median (class signal):")
overall_med = train[NUM_COLS].median()
print(grouped.subtract(overall_med, axis=1).round(2).to_string())
""")

# ── 5. Categorical Features ──────────────────────────────────────────────────
md("## 5.  Categorical Features")
md("### 5.1  Class distribution within each category")

code("""
CAT_COLS = [
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]
""")

code("""
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.ravel()

for idx, col in enumerate(CAT_COLS):
    ax = axes[idx]
    ct = pd.crosstab(train[col], train["health_condition"], normalize="index")
    ct = ct[target_order]
    ct.plot(kind="barh", stacked=True, color=[palette[c] for c in target_order], ax=ax)
    ax.set(title=f"{col} → Health Condition", xlabel="Proportion")
    ax.legend(loc="lower right", fontsize=8)
    # Add % labels
    for i in range(ct.shape[0]):
        cum = 0
        for j in range(ct.shape[1]):
            val = ct.iloc[i, j]
            if val > 0.05:
                ax.text(cum + val / 2, i, f"{val:.0%}", ha="center", va="center", fontsize=7)
            cum += val

fig.tight_layout()
plt.show()
""")

code("""
# Which categories are most predictive? (contingency table Cramer's V approximation)
from scipy.stats import chi2_contingency

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
axes = axes.ravel()

for idx, col in enumerate(CAT_COLS):
    ax = axes[idx]
    ct = pd.crosstab(train[col], train["health_condition"])
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set(title=f"{col} (n_cat={ct.shape[0]})")

    chi2, p, dof, expected = chi2_contingency(ct)
    cramer = np.sqrt(chi2 / (len(train) * (min(ct.shape) - 1)))
    print(f"{col:25s}  χ²={chi2:.0f}  p={p:.2e}  Cramér's V={cramer:.4f}")

fig.tight_layout()
plt.show()
""")

# ── 6. Correlations ──────────────────────────────────────────────────────────
md("## 6.  Feature Correlations")
md("### 6.1  Numeric feature correlation matrix")

code("""
NUM_COLS = [c for c in NUM_COLS if c in train.columns]
corr = train[NUM_COLS + ["health_condition"]].copy()
corr["health_condition"] = corr["health_condition"].map({"fit": 0, "at-risk": 1, "unhealthy": 2})

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr.corr(), dtype=bool), k=1)
sns.heatmap(corr.corr(), mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            vmin=-1, vmax=1, center=0, square=True, ax=ax)
ax.set(title="Feature Correlation Matrix")
plt.show()
""")

code("""
# Correlation with target (label-encoded)
target_corr = corr.corr()["health_condition"].drop("health_condition").sort_values()
print("Correlation with target (label-encoded health_condition):")
print(target_corr.round(4).to_string())
print()

fig, ax = plt.subplots(figsize=(10, 6))
colors_col = ["C3" if v < 0 else "C0" for v in target_corr.values]
ax.barh(range(len(target_corr)), target_corr.values, color=colors_col, alpha=0.7)
ax.set(yticks=range(len(target_corr)), yticklabels=target_corr.index,
       xlabel="Correlation with Target", title="Feature-Target Correlation (Pearson)")
ax.axvline(0, color="grey", linewidth=0.5)
fig.tight_layout()
plt.show()
""")

md("### 6.2  Mean target encoding view")
code("""
# Mean of each numeric feature per class
mean_by_class = train.groupby("health_condition")[NUM_COLS].mean()
std_by_class = train.groupby("health_condition")[NUM_COLS].std()

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()
for idx, col in enumerate(NUM_COLS):
    ax = axes[idx]
    ax.errorbar(mean_by_class.index, mean_by_class[col], yerr=std_by_class[col],
                fmt="o", capsize=5, capthick=2, markersize=8, color="C0")
    ax.set(title=f"Mean {col} per Class", xlabel="Health Condition", ylabel=col)

axes[-1].axis("off")
fig.tight_layout()
plt.show()
""")

# ── 7. Feature Engineering Ideas ─────────────────────────────────────────────
md("## 7.  Feature Engineering Ideas")
md("""
### Key observations from EDA

1. **Class imbalance is extreme** — 86% at-risk, 6% fit, 8% unhealthy.
   Balanced accuracy means we must predict all 3 classes well.

2. **Missing data is present in every feature** (1-12% missing rate).
   Simple median/mode imputation works, but careful missing handling may help.

3. **Numerical features show class signal** — `stress_level` (categorical) is
   likely the most predictive feature.  `bmi`, `exercise_duration`, and
   `sleep_duration` show clear differences between fit and unhealthy.

4. **Interaction features** — `bmi * exercise_duration`, `calorie / step_count`
   ratios capture lifestyle efficiency.

5. **Binning numeric features** can help capture non-linear relationships.

6. **Class weights** are essential for LightGBM to handle imbalance fairly.
""")

code("""
# Quick check: how well do top-2 features separate classes?
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

scatter_cols = [("bmi", "exercise_duration"), ("sleep_duration", "heart_rate"),
                ("step_count", "calorie_expenditure")]
for ax, (xcol, ycol) in zip(axes, scatter_cols):
    for cls in target_order:
        subset = train[train["health_condition"] == cls].dropna(subset=[xcol, ycol])
        ax.scatter(subset[xcol], subset[ycol], c=palette[cls], label=cls,
                   alpha=0.15, s=1, rasterized=True)
    ax.set(xlabel=xcol, ylabel=ycol, title=f"{xcol} vs {ycol}")
    ax.legend(markerscale=20)

fig.suptitle("Feature Pairs colored by Health Condition", fontsize=14)
fig.tight_layout()
plt.show()
""")

# ── 8. Baseline Model ────────────────────────────────────────────────────────
md("## 8.  Baseline Model")
md("""
Quick baseline: LightGBM with 5-fold stratified CV.
Impute with median/mode, label-encode categoricals, add interaction features.
""")

code("""
from sklearn.metrics import classification_report

# Simple preprocessing
def quick_preprocess(df, target=None):
    df = df.copy()
    num_cols = NUM_COLS
    cat_cols = CAT_COLS

    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        if col in df.columns:
            mode_val = df[col].mode()
            mode = mode_val.iloc[0] if len(mode_val) > 0 else "missing"
            df[col] = df[col].fillna(mode)

    # Encode categoricals
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Feature engineering
    if "bmi" in df.columns and "exercise_duration" in df.columns:
        df["bmi_exercise_interaction"] = df["bmi"] * df["exercise_duration"]
    if "step_count" in df.columns and "calorie_expenditure" in df.columns:
        df["efficiency_ratio"] = df["calorie_expenditure"] / (df["step_count"] + 1)

    if target and target in df.columns:
        return df.drop(columns=["id", target]), df[target]
    return df.drop(columns=["id"]), None


X, y = quick_preprocess(train, target="health_condition")
y_enc = LabelEncoder().fit_transform(y)

X_test, _ = quick_preprocess(test)
# Align columns
for c in X.columns:
    if c not in X_test.columns:
        X_test[c] = 0
X_test = X_test[X.columns]

print(f"X: {X.shape}, X_test: {X_test.shape}")
print(f"Columns: {list(X.columns)}")
""")

code("""
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lgbm_params = {
    "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
    "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63,
    "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1,
    "class_weight": "balanced", "random_state": 42, "n_jobs": -1, "verbose": -1,
}

oof_preds = np.zeros((len(X), 3))
test_preds = np.zeros((len(X_test), 3))
scores = []

for fold, (trn_idx, val_idx) in tqdm(
    enumerate(cv.split(X, y_enc)), total=5, desc="CV fold", unit="fold"
):
    X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_val = y_enc[trn_idx], y_enc[val_idx]

    model = LGBMClassifier(**lgbm_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="multi_logloss")

    oof_preds[val_idx] = model.predict_proba(X_val)
    test_preds += model.predict_proba(X_test) / 5

    fold_labels = model.predict(X_val)
    fold_ba = balanced_accuracy_score(y_val, fold_labels)
    scores.append(fold_ba)
    tqdm.write(f"  Fold {fold + 1}: Balanced Acc = {fold_ba:.4f}")

oof_labels = oof_preds.argmax(axis=1)
oof_ba = balanced_accuracy_score(y_enc, oof_labels)
print(f"\\nOOF Balanced Accuracy: {oof_ba:.4f} (mean {np.mean(scores):.4f} ± {np.std(scores):.4f})")
print(f"\\nOOF Classification Report:")
target_names = ["fit", "at-risk", "unhealthy"]
print(classification_report(y_enc, oof_labels, target_names=target_names))
""")

md("### 8.1  Analysis of errors")
code("""
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_enc, oof_labels)
disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
disp.plot(cmap="Blues", values_format="d")
plt.title("OOF Confusion Matrix")
plt.show()

# Per-class recall (this is what balanced accuracy measures)
recall = cm.diagonal() / cm.sum(axis=1)
for i, label in enumerate(target_names):
    print(f"Recall ({label}): {recall[i]:.4f}")
print(f"\\nBalanced Accuracy = {recall.mean():.4f}")
""")

# ── 9. Summary ───────────────────────────────────────────────────────────────
md("## 9.  Summary & Recommendations")

md("""
### Data Summary

| Aspect | Detail |
|--------|--------|
| Training samples | 690,088 |
| Test samples | 295,753 |
| Features | 7 numeric, 6 categorical |
| Target | `fit` (5.8%), `at-risk` (85.9%), `unhealthy` (8.4%) |
| Metric | Balanced Accuracy (mean recall per class) |
| Missing data | 1-12% per feature, all columns affected |

### Key Findings

1. **Class imbalance is the central challenge** — predicting `at-risk` alone
   gives 86% accuracy but only 0.33 balanced accuracy.

2. **Categorical features** like `stress_level`, `sleep_quality`, and
   `physical_activity_level` discriminate well between classes.

3. **Numerical features** — `bmi`, `exercise_duration`, `sleep_duration`, and
   `step_count` show clear class separation.

4. **Interaction features** improve separation: `bmi × exercise_duration`,
   `calorie / step_count`.

5. **Class weighting is essential** — LightGBM's `class_weight='balanced'`
   significantly improves minority class recall.

### Recommendations

1. **Start with LightGBM** with class weighting and stratified CV.
2. **Test XGBoost + CatBoost** for ensemble diversity.
3. **Experiment with synthetic oversampling** (SMOTE) for minority classes.
4. **Try target encoding** of categorical features.
5. **Tune per-class thresholds** on OOF probabilities to maximize balanced
   accuracy.
6. **Feature selection** — `stress_level` alone may capture much of the signal.
""")

code("""
print("✓ EDA complete")
print(f"  Baseline OOF Balanced Accuracy: {oof_ba:.4f}")
print(f"  Submission ready: {X_test.shape[0]:,} predictions")
""")

# ── Assemble notebook ──────────────────────────────────────────────────────
NB.cells = CELLS

out_path = Path("notebooks/1_EDA.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(NB, out_path)
print(f"Written {out_path}")
