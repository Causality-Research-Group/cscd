"""Corrected decision task (Table 27, answer to Rfqt-Q3) from rerun records.

Usage:  python -m rebuttal.decision_eval RECORD_DIR
Compares the corrected rerun with the committed (buggy) per-repeat summary.

Conventions:
* every set is evaluated on the same repeats;
* an empty CSCD set is an abstention: it gives no estimate and does not cover;
  its abstention rate is reported, and bias is reported over non-abstaining repeats;
* coverage is the fraction of (treatment, outcome) pairs whose true effect lies in
  [min, max] of all per-graph local-IDA estimates of the set, averaged over repeats.
"""

import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

SETS = ["single", "cscd_set", "pool", "bootstrap_avg", "bma"]
OLD = "results_v3/summary/decision_per_repeat.csv"


def load(record_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(record_dir, "rec_*.pkl"))):
        with open(path, "rb") as f:
            rec = pickle.load(f)
        dec = rec.get("decision")
        if not dec:
            continue
        row = dict(method=rec["method"], repeat=rec["repeat"],
                   size_cscd=dec.get("dec2_size_cscd_set", np.nan), size_pool=dec.get("dec2_size_pool", np.nan))
        for s in SETS:
            row[f"bias_{s}"] = dec.get(f"dec2_bias_{s}", np.nan)
            row[f"cover_{s}"] = dec.get(f"dec2_cover_{s}", np.nan)
            row[f"width_{s}"] = dec.get(f"dec2_width_{s}", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df, label):
    out = []
    for method, g in df.groupby("method"):
        abstain = g.size_cscd == 0 if "size_cscd" in g else pd.Series(False, index=g.index)
        row = {"version": label, "method": method, "repeats": len(g)}
        if "size_cscd" in g:
            row["CSCD abstains"] = f"{abstain.mean():.2f}"
            row["CSCD set / pool size"] = f"{g.size_cscd.mean():.0f} / {g.size_pool.mean():.0f}"
        for s in SETS:
            cover = g[f"cover_{s}"].copy()
            if s == "cscd_set":
                cover = cover.where(~abstain, 0.0)  # abstention: no coverage
            row[f"cover {s}"] = f"{cover.mean():.2f}"
            row[f"bias {s}"] = f"{g[f'bias_{s}'].mean():.3f}"
        out.append(row)
    return pd.DataFrame(out)


def subset_check(df):
    """The corrected CSCD range is inside the pool's, so its coverage and width cannot exceed the pool's."""
    ok = df.dropna(subset=["width_cscd_set", "width_pool"])
    width = bool((ok.width_cscd_set <= ok.width_pool + 1e-9).all())
    cover = bool((ok.cover_cscd_set <= ok.cover_pool + 1e-9).all())
    return width, cover


def main(record_dir):
    new = load(record_dir)
    old = pd.read_csv(OLD).rename(columns=lambda c: c.replace("dec2_", ""))
    table = pd.concat([summarize(old, "submitted (bug)"), summarize(new, "corrected")], ignore_index=True)
    os.makedirs("rebuttal/out", exist_ok=True)
    table.to_csv("rebuttal/out/decision_corrected.csv", index=False)
    print(table.T.to_string())
    width_ok, cover_ok = subset_check(new)
    print("\ncorrected: CSCD width <= pool width in every repeat:", width_ok)
    print("corrected: CSCD coverage <= pool coverage in every repeat:", cover_ok)
    old_bad = (old.cover_cscd_set > old.cover_pool + 1e-9).sum()
    print("submitted: repeats where CSCD coverage > pool coverage:", int(old_bad), "of", len(old))
    # Columns other than the CSCD set must be unchanged by the fix (same seeds), except "pool",
    # which is now the 101-graph standard family instead of all 207 standard-tagged graphs.
    merged = old.merge(new, on=["method", "repeat"], suffixes=("_old", "_new"))
    for s in ("single", "bootstrap_avg", "bma"):
        diff = np.nanmax(np.abs(merged[f"bias_{s}_old"] - merged[f"bias_{s}_new"]))
        print(f"max |bias change| for {s} (should be 0): {diff:.2e}")


if __name__ == "__main__":
    main(sys.argv[1])
