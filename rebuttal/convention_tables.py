"""Gamma and sample-size tables under one precision convention (answer to AI-3.5(b)).

Usage:  python -m rebuttal.convention_tables

The v2 tables (tab:gamma, tab:samplesize) averaged precision with a mixed
rule: a repeat whose passing set was empty was left out (NaN), but a repeat
with a non-empty set and no accepted claim counted as precision 0.  Here a
repeat with no accepted claim *abstains*: it is left out of the precision
average and counted in a separate "Abstain" column.  Recall is unchanged
(an abstaining repeat has recall 0).

An adjacency is accepted exactly when one of its two directed claims is
(skeleton frequency = max of the two directed frequencies), so the stored
number of accepted directed claims, ``cscd_n_accepted``, tells whether a
repeat accepted any claim of either type.

Reads results_v2/<exp>/metrics.csv; writes rebuttal/out/gamma.tex and
rebuttal/out/samplesize.tex.
"""

import os

import numpy as np
import pandas as pd

ROOT = "results_v2"
OUT = "rebuttal/out"
ORDER = ["pc", "ges", "fci"]
LABEL = {"pc": "PC", "ges": "GES", "fci": "FCI"}


def load(exp: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(ROOT, exp, "metrics.csv"))
    return df[(df.distance == df.distance.iloc[0]) & np.isclose(df.alpha, 0.1)]


def mean_se(x: pd.Series, dec: int) -> str:
    x = x.dropna()
    if len(x) == 0:
        return "--"
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return f"${x.mean():.{dec}f}\\pm{se:.{dec}f}$"


def summary(d: pd.DataFrame) -> dict:
    """Precision over repeats with a claim; recall and abstention over all repeats."""
    claimed = d.cscd_n_accepted.fillna(0) > 0
    return dict(
        adj_prec=mean_se(d.cscd_skprec[claimed], 3), adj_rec=mean_se(d.cscd_skrec.fillna(0), 3),
        dir_prec=mean_se(d.cscd_precision[claimed], 3), dir_rec=mean_se(d.cscd_recall.fillna(0), 3),
        unresolved=mean_se(d.cscd_unresolved, 3), abstain=f"${(~claimed).mean():.2f}$",
        old_adj_prec=d.cscd_skprec.mean(),
    )


def gamma_table() -> list:
    df = load("main_v2")
    lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
             r"Method & $\gamma$ & Adj.\ prec. & Adj.\ rec. & Dir.\ prec. & Dir.\ rec. & Unresolved & Abstain \\",
             r"\midrule"]
    rows = []
    for g in (0.6, 0.7, 0.8, 0.9, 1.0):
        for m in ORDER:
            s = summary(df[np.isclose(df.gamma, g) & (df.method == m)])
            rows.append(dict(method=m, gamma=g, **s))
            lines.append(f"{LABEL[m]} & {g} & {s['adj_prec']} & {s['adj_rec']} & {s['dir_prec']} & {s['dir_rec']} & "
                         f"{s['unresolved']} & {s['abstain']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT, "gamma.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return rows


def samplesize_table() -> list:
    settings = [(f"n{n}_N9", n) for n in (200, 400, 800, 1600)] + [("design_N9", 3200), ("n6400_N9", 6400),
                                                                   ("n12800_N9", 12800)]
    lines = [r"\begin{tabular}{llccccccc}", r"\toprule",
             r"Setting & Method & Exceed. & Admitted & $G^\ast$ incl. & Adj.\ prec. & Adj.\ rec. & Dir.\ prec. & Abstain \\",
             r"\midrule"]
    rows = []
    for exp, n in settings:
        df = load(exp)
        df = df[np.isclose(df.gamma, 0.9)]
        first = True
        for m in ORDER:
            d = df[df.method == m]
            if d.empty:
                continue
            s = summary(d)
            rows.append(dict(n=n, method=m, **s))
            lines.append(f"{f'$n{{=}}{n}$' if first else ''} & {LABEL[m]} & {mean_se(d.exceedance, 3)} & "
                         f"{mean_se(d.admission, 2)} & {mean_se(d.true_included, 2)} & {s['adj_prec']} & "
                         f"{s['adj_rec']} & {s['dir_prec']} & {s['abstain']} \\\\")
            first = False
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT, "samplesize.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(gamma_table()).to_string())
    print(pd.DataFrame(samplesize_table())[["n", "method", "adj_prec", "dir_prec", "abstain", "old_adj_prec"]].to_string())
