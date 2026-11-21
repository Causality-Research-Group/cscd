"""LaTeX body of the corrected decision-task table (paper Table 27, tables/rev/decision.tex).

Usage:  python -m rebuttal.decision_table RECORD_DIR OUT_TEX

Conventions (as in rebuttal/decision_eval.py):
* every set is evaluated on the same repeats and the same (treatment, outcome) pairs;
* coverage of a repeat is the fraction of its pairs whose true effect lies in the
  [min, max] range of the set's local-IDA estimates; an empty CSCD set gives no
  estimate and counts as coverage 0;
* bias is the mean absolute error of the set-averaged estimate, over the repeats in
  which the set gives an estimate.
Entries are means over repeats plus or minus one standard error.
"""

import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

COLUMNS = [("oracle", "Oracle"), ("naive", "None"), ("single", "Single"), ("cscd_set", "CSCD set"),
           ("pool", "Pool"), ("bootstrap_avg", "Bootstrap"), ("bma", "BMA")]
METHODS = [("pc", "PC"), ("ges", "GES"), ("fci", "FCI")]


def load(record_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(record_dir, "rec_*.pkl"))):
        with open(path, "rb") as f:
            rec = pickle.load(f)
        dec = rec.get("decision")
        if not dec:
            continue
        row = {"method": rec["method"], "repeat": rec["repeat"], "size_cscd": dec.get("dec2_size_cscd_set", np.nan)}
        for key, _ in COLUMNS:
            row[f"bias_{key}"] = dec.get(f"dec2_bias_{key}", np.nan)
            row[f"cover_{key}"] = dec.get(f"dec2_cover_{key}", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def cell(values, digits):
    v = pd.Series(values).dropna()
    if v.empty:
        return "--"
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return f"${v.mean():.{digits}f}\\pm{se:.{digits}f}$"


def main(record_dir, out_tex):
    df = load(record_dir)
    lines = ["\\begin{tabular}{lccccccc}", "\\toprule",
             "Method & " + " & ".join(label for _, label in COLUMNS) + " \\\\", "\\midrule",
             "\\multicolumn{8}{l}{\\emph{Mean absolute bias}} \\\\"]
    for m, label in METHODS:
        g = df[df.method == m]
        lines.append(label + " & " + " & ".join(cell(g[f"bias_{k}"], 3) for k, _ in COLUMNS) + " \\\\")
    lines += ["\\midrule", "\\multicolumn{8}{l}{\\emph{Coverage of the true effect by the range of estimates}} \\\\"]
    for m, label in METHODS:
        g = df[df.method == m]
        cells = []
        for k, _ in COLUMNS:
            if k in ("oracle", "naive"):
                cells.append("--")
                continue
            cover = g[f"cover_{k}"]
            if k == "cscd_set":
                cover = cover.where(g.size_cscd > 0, 0.0)  # abstention: no coverage
            cells.append(cell(cover, 2))
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
