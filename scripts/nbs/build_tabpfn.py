"""Build notebooks/3_TabPFN.ipynb programmatically via nbformat.

Usage:
    uv run python scripts/nbs/build_tabpfn.py
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


# ── %cd .. ──
code("""%cd ..
""")

# ── Title ──
md("""
# Student Health Risk — TabPFN v3

**TabPFN** is a foundation model for tabular data — it uses in-context learning
via a transformer trained on millions of synthetic datasets.  No hyperparameter
tuning, no feature engineering, no imputation needed.

[TabPFN on GitHub](https://github.com/priorlabs/tabpfn) |
[Paper](https://arxiv.org/abs/2407.13271)

> **Idea**: Let a foundation model (trained on 10M+ synthetic tabular tasks)
> classify student health risk.  TabPFN handles missing values, categoricals,
> and class imbalance without any manual preprocessing.
""")

# ── 1. Setup ──
md("## 1. Setup")
code("""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tabpfn import TabPFNClassifier

warnings.filterwarnings("ignore")

NUM_COLS = [
    "sleep_duration", "heart_rate", "bmi", "calorie_expenditure",
    "step_count", "exercise_duration", "water_intake",
]
CAT_COLS = [
    "diet_type", "stress_level", "sleep_quality",
    "physical_activity_level", "smoking_alcohol", "gender",
]
TARGET = "health_condition"
TARGET_MAP = {"fit": 0, "at-risk": 1, "unhealthy": 2}

_DATA_CANDIDATES = [Path("data/raw"), Path("../data/raw")]
DATA_DIR = next((p for p in _DATA_CANDIDATES if (p / "train.csv").exists()), _DATA_CANDIDATES[0])
print(f"✓ DATA_DIR = {DATA_DIR.resolve()}")
""")

# ── 2. Load ──
md("## 2. Load Data")
code("""
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
test_ids = test["id"].copy()

print(f"Train: {train.shape}  |  Test: {test.shape}")
print(f"Target:\\n{train[TARGET].value_counts()}")
""")

# ── 3. Prepare ──
md("## 3. Prepare Features")
md("""
TabPFN handles **missing values natively** — no imputation.
Categoricals are passed as strings; TabPFN encodes them internally.

We **stratified subsample** to 15K rows because TabPFN's in-context learning
sweet spot is 1K–15K samples.  The subsample preserves class proportions.
""")

code("""
SAMPLE_SIZE = 15000
sss = StratifiedShuffleSplit(n_splits=1, train_size=SAMPLE_SIZE,
                              random_state=42)
idx, _ = next(sss.split(train, train[TARGET]))
sample = train.iloc[idx].reset_index(drop=True)

print(f"Sampled {len(sample):,} rows")
print(f"Distribution:\\n{sample[TARGET].value_counts()}")

feature_cols = NUM_COLS + CAT_COLS
X_tr = sample[feature_cols].copy()
y_tr = sample[TARGET].map(TARGET_MAP).values

X_te = test[feature_cols].copy()

# Ensure dtypes
for c in NUM_COLS:
    X_tr[c] = X_tr[c].astype("float32")
    X_te[c] = X_te[c].astype("float32")
for c in CAT_COLS:
    X_tr[c] = X_tr[c].astype(str)
    X_te[c] = X_te[c].astype(str)

cat_indices = list(range(len(NUM_COLS), len(feature_cols)))
print(f"Categorical column indices: {cat_indices}")
print(f"X_tr: {X_tr.shape}  X_te: {X_te.shape}")
""")

# ── 4. Train TabPFN ──
md("## 4. Train TabPFN")
md("""
TabPFN is a transformer — it "trains" by simply encoding the training data
as context.  The forward pass runs on GPU if available.
""")

code("""
print("Fitting TabPFN ...")
model = TabPFNClassifier(
    categorical_features_indices=cat_indices,
    n_estimators=8,
    random_state=42,
    show_progress_bar=False,
)
model.fit(X_tr, y_tr)
print("✓ Fit complete")
""")

# ── 5. Predict ──
md("## 5. Predict & Submit")
code("""
preds = model.predict_proba(X_te)
test_labels = preds.argmax(axis=1)
rev_map = {v: k for k, v in TARGET_MAP.items()}
pred_labels = [rev_map[i] for i in test_labels]

print("Predicted distribution:")
for label in ["fit", "at-risk", "unhealthy"]:
    pct = 100 * pred_labels.count(label) / len(pred_labels)
    print(f"  {label}: {pred_labels.count(label)} ({pct:.1f}%)")
""")

code("""
out_dir = Path("data/submissions")
out_dir.mkdir(parents=True, exist_ok=True)
submission = pd.DataFrame({"id": test_ids, "health_condition": pred_labels})
submission.to_csv(out_dir / "submission.csv", index=False)
print(f"Saved to {out_dir / 'submission.csv'}")
print(f"Shape: {submission.shape}")
submission.head()
""")

# ── 6. Feature importance ──
md("## 6. Feature Importance (TabPFN explanation)")
md("""
TabPFN doesn't provide traditional feature importance, but we can estimate
it by measuring the drop in balanced accuracy when shuffling each feature.
""")

code("""
from sklearn.metrics import balanced_accuracy_score

y_pred = model.predict(X_tr)
base_score = balanced_accuracy_score(y_tr, y_pred)
print(f"Baseline balanced acc (in-sample): {base_score:.4f}")

importance = {}
for i, col in enumerate(feature_cols):
    X_perm = X_tr.copy()
    X_perm.iloc[:, i] = np.random.permutation(X_perm.iloc[:, i].values)
    y_perm = model.predict(X_perm)
    drop = base_score - balanced_accuracy_score(y_tr, y_perm)
    importance[col] = drop

imp_df = pd.Series(importance).sort_values(ascending=False)
print("\\nFeature importance (drop in balanced acc when shuffled):")
for col, val in imp_df.items():
    bar = "█" * int(abs(val) * 200)
    print(f"  {col:25s} {val:+.4f}  {bar}")
""")

# ── 7. Summary ──
md("## 7. Summary")
md("""
### What TabPFN brings

1. **Zero preprocessing** — no imputation, no encoding, no scaling
2. **No tuning** — foundation model works out of the box
3. **Strong baseline** — often beats tuned tree ensembles on small–medium data
4. **GPU accelerated** — runs on Kaggle's T4 GPU

### Limitations

- 15K subsample means we discard 98% of training data
- No native feature importance (permutation-based is slow)
- Larger ensemble (e.g., `n_estimators=32`) improves score but costs more

### Next steps to improve

1. Ensemble TabPFN with LightGBM (different inductive biases)
2. Multiple subsample runs averaged (reduce variance)
3. Increase `n_estimators` to 16–32 if GPU memory allows
4. Try `ignore_pretraining_limits=True` with more samples
""")

code("""
print("✓ TabPFN baseline complete")
print(f"  Submission: data/submissions/submission.csv")
""")

# ── Assemble ──
NB.cells = CELLS

out_path = Path("notebooks/3_TabPFN.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(NB, out_path)
print(f"Written {out_path}")
