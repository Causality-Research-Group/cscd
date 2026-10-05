"""Aggregate revision-experiment records into per-repeat metrics and summaries.

Usage:
    python analyze_v2.py [experiment ...]      # default: every folder in results_v2

Writes results_v2/<exp>/metrics.csv (one row per repeat, method, alpha,
gamma, distance) and results_v2/summary/<exp>.csv (mean, standard error and
count per method at each setting).  Summaries are small and are committed.
"""

import glob
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

from cscd.analysis import record_metrics

ROOT = "results_v2"
ALPHAS = (0.05, 0.1, 0.2)
GAMMAS = (0.6, 0.7, 0.8, 0.9, 1.0)


def analyse(exp: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, exp, "rec_*.pkl"))):
        with open(path, "rb") as f:
            rec = pickle.load(f)
        distances = list(rec["cal_scores"].keys())
        for d in distances:
            for a in ALPHAS:
                for g in (GAMMAS if (d == distances[0] and a == 0.1) else (0.9,)):
                    m = record_metrics(rec, alpha=a, gamma=g, distance=d)
                    m.update(experiment=exp, method=rec["method"], repeat=rec["repeat"],
                             alpha=a, gamma=g, distance=d)
                    rows.append(m)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["experiment", "method", "distance", "alpha", "gamma"]
    num = [c for c in df.columns if c not in keys + ["repeat"] and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(keys)[num]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean = g.mean().add_suffix("_mean")
        se = (g.std(ddof=1) / np.sqrt(g.count())).add_suffix("_se")
    count = df.groupby(keys)["repeat"].nunique().rename("n_repeats")
    return pd.concat([mean, se, count], axis=1).reset_index()


def main() -> None:
    exps = sys.argv[1:] or sorted(
        e for e in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, e)) and e != "summary"
    )
    os.makedirs(os.path.join(ROOT, "summary"), exist_ok=True)
    for exp in exps:
        df = analyse(exp)
        if df.empty:
            print(f"[skip] {exp}: no records")
            continue
        df.to_csv(os.path.join(ROOT, exp, "metrics.csv"), index=False)
        summarise(df).to_csv(os.path.join(ROOT, "summary", f"{exp}.csv"), index=False)
        print(f"{exp}: {df['repeat'].nunique()} repeats x {df['method'].nunique()} methods")


if __name__ == "__main__":
    main()
