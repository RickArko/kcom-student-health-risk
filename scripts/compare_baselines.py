#!/usr/bin/env python3
"""Compare ensemble OOF BA against AutoGluon / TabPFN verification baselines.

Exit codes:
  0 — blend beats AutoGluon (or AutoGluon missing) and meets optional floor
  1 — blend loses to AutoGluon
  2 — ensemble metrics missing

Usage:
    uv run python scripts/compare_baselines.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_ba(path: Path) -> float | None:
    metrics = path / "metrics.json"
    if not metrics.exists():
        # nested blend/ layout from save_oof_artifacts
        nested = path / "blend" / "metrics.json"
        if nested.exists():
            metrics = nested
        else:
            return None
    data = json.loads(metrics.read_text())
    for key in ("oof_ba", "oof_ba_tuned"):
        if key in data:
            return float(data[key])
    return None


def _load_per_model(path: Path) -> dict[str, float]:
    metrics = path / "metrics.json"
    if not metrics.exists():
        return {}
    data = json.loads(metrics.read_text())
    per = data.get("per_model") or {}
    return {k: float(v) for k, v in per.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare ensemble vs AutoGluon/TabPFN")
    p.add_argument("--ensemble-dir", type=Path, default=Path("experiments/ensemble"))
    p.add_argument("--autogluon-dir", type=Path, default=Path("experiments/autogluon"))
    p.add_argument("--tabpfn-dir", type=Path, default=Path("experiments/tabpfn"))
    p.add_argument("--min-ba", type=float, default=0.94, help="Soft floor for ensemble OOF BA")
    p.add_argument(
        "--require-autogluon",
        action="store_true",
        help="Fail if AutoGluon metrics are missing",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[tuple[str, float | None]] = []

    per_model = _load_per_model(args.ensemble_dir)
    for name, ba in sorted(per_model.items()):
        rows.append((name, ba))

    blend_ba = _load_ba(args.ensemble_dir)
    rows.append(("blend", blend_ba))
    ag_ba = _load_ba(args.autogluon_dir)
    rows.append(("autogluon", ag_ba))
    tab_ba = _load_ba(args.tabpfn_dir)
    rows.append(("tabpfn", tab_ba))

    print()
    print(f"{'model':<12}  {'oof_ba':>8}")
    print("-" * 24)
    for name, ba in rows:
        print(f"{name:<12}  {ba:>8.4f}" if ba is not None else f"{name:<12}  {'n/a':>8}")
    print()

    if blend_ba is None:
        print("ERROR: ensemble metrics not found. Run: make train-ensemble")
        sys.exit(2)

    if blend_ba < args.min_ba:
        print(f"WARNING: blend OOF BA {blend_ba:.4f} < soft floor {args.min_ba:.4f}")

    if ag_ba is None:
        msg = "AutoGluon metrics missing (run: make verify-autogluon)"
        if args.require_autogluon:
            print(f"ERROR: {msg}")
            sys.exit(2)
        print(f"NOTE: {msg} — skipping AutoGluon gate")
    elif blend_ba + 1e-12 < ag_ba:
        print(f"FAIL: blend ({blend_ba:.4f}) < AutoGluon ({ag_ba:.4f})")
        sys.exit(1)
    else:
        print(f"PASS: blend ({blend_ba:.4f}) >= AutoGluon ({ag_ba:.4f})")

    if tab_ba is not None:
        if blend_ba + 1e-12 >= tab_ba:
            print(f"PASS: blend ({blend_ba:.4f}) >= TabPFN ({tab_ba:.4f})")
        else:
            print(
                f"NOTE: blend ({blend_ba:.4f}) < TabPFN ({tab_ba:.4f}) — "
                "consider blending TabPFN OOF"
            )

    print("Done.")
    sys.exit(0)


if __name__ == "__main__":
    main()
