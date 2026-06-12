#!/usr/bin/env python3
"""
make_paper_artifacts.py
=======================
Aggregate experiment results and produce paper-ready LaTeX tables, figures,
a summary digest CSV, and a plain-text CHECKS section.

Usage
-----
    python make_paper_artifacts.py [--results_root results] [--outdir artifacts]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALPHA_LEVELS = [0.05, 0.10, 0.20]
ALPHA_10 = 0.10

METHOD_ORDER = ["pc", "ges", "fci"]
METHOD_LABELS = {"pc": "PC", "ges": "GES", "fci": "FCI"}

REGIME_ORDER = ["none", "correct+hard", "correct+soft", "noisy+hard", "noisy+soft"]

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})


def se(series: pd.Series) -> float:
    """Standard error, ignoring NaN."""
    n = series.dropna().count()
    if n <= 1:
        return float("nan")
    return series.std(ddof=1) / math.sqrt(n)


def fmt_pm(mean: float, stderr: float, decimals: int = 3) -> str:
    """Format as $mean \\pm se$ for LaTeX."""
    if math.isnan(mean):
        return r"\text{--}"
    if math.isnan(stderr):
        return f"${mean:.{decimals}f}$"
    return f"${mean:.{decimals}f} \\pm {stderr:.{decimals}f}$"


def fmt_single(value: float, decimals: int = 3) -> str:
    """Format a single value for LaTeX."""
    if math.isnan(value):
        return r"\text{--}"
    return f"${value:.{decimals}f}$"


def load_csv(path: str) -> pd.DataFrame | None:
    """Load a CSV, returning None and printing a warning if missing."""
    if not os.path.isfile(path):
        print(f"  [WARNING] Missing input file: {path} — skipping.", flush=True)
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception as exc:
        print(f"  [WARNING] Failed to read {path}: {exc} — skipping.", flush=True)
        return None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def agg_group(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    """
    Aggregate value_cols by group_cols, computing mean and SE.
    Returns wide dataframe with columns like col_mean, col_se.
    """
    agg_dict = {c: ["mean", se] for c in value_cols if c in df.columns}
    if not agg_dict:
        return pd.DataFrame()
    agg = df.groupby(group_cols, as_index=False).agg(agg_dict)
    # Flatten MultiIndex columns
    agg.columns = [
        "_".join(str(s) for s in col).rstrip("_") if isinstance(col, tuple) else col
        for col in agg.columns
    ]
    # Rename 'se' suffix (pandas uses the function name)
    agg.columns = [c.replace("_<lambda>", "_se") for c in agg.columns]
    # If pandas named it differently (function name 'se')
    agg.columns = [c.replace("_se_0", "_se") for c in agg.columns]
    return agg


def _agg_with_se(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    """
    Robust aggregation that returns mean and SE for each value column.
    Works around pandas function-naming quirks.
    """
    records = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict = dict(zip(group_cols, keys))
        for col in value_cols:
            if col not in sub.columns:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_se"] = float("nan")
            else:
                vals = sub[col].dropna()
                n = len(vals)
                m = vals.mean() if n > 0 else float("nan")
                s = (vals.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
                row[f"{col}_mean"] = m
                row[f"{col}_se"] = s
        records.append(row)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

BOOKTABS_HEADER = r"""\begin{table}[t]
\centering
\small
\begin{tabular}{%s}
\toprule
%s \\
\midrule
"""

BOOKTABS_FOOTER = r"""\bottomrule
\end{tabular}
\caption{%s}
\label{%s}
\end{table}
"""


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


# ---------------------------------------------------------------------------
# Artifact 2a: table_synthetic_main.tex
# ---------------------------------------------------------------------------

def make_table_synthetic_main(df: pd.DataFrame, outpath: str) -> None:
    """
    One block per method (PC, GES, FCI).
    Columns: Empirical exceedance at alpha=.05/.10/.20, Self-compatibility,
             Accepted precision, Accepted recall (all at alpha=.10 except exceedance),
             Unresolved rate, Skeleton F1, SHD (full_graph_shd).
    Values: mean +- se.
    """
    cols_needed = [
        "cal_exceed_rate", "raw_self_compatibility",
        "accepted_precision", "accepted_recall",
        "unresolved_rate", "skeleton_f1", "full_graph_shd",
    ]

    # 10 columns: Method | (blank) | 3 exceedance | self-comp | prec | rec | unres | f1 | shd
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{llccccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Method & & Exc.\ $\alpha{=}.05$ & Exc.\ $\alpha{=}.10$ & Exc.\ $\alpha{=}.20$ "
        r"& Self-comp.\ & Acc.\ Prec. & Acc.\ Rec. & Unres. Rate & Skel.\ F1 & SHD \\"
    )
    lines.append(r"\midrule")

    methods_present = [m for m in METHOD_ORDER if m in df["method"].values]

    for i, method in enumerate(methods_present):
        sub = df[df["method"] == method]
        label = METHOD_LABELS.get(method, method.upper())

        # Exceedance at the three alphas
        exc_vals = {}
        for alpha in ALPHA_LEVELS:
            sub_a = sub[sub["alpha"].round(3) == round(alpha, 3)]
            if sub_a.empty:
                exc_vals[alpha] = (float("nan"), float("nan"))
            else:
                vals = sub_a["cal_exceed_rate"].dropna()
                n = len(vals)
                m = vals.mean() if n > 0 else float("nan")
                s = (vals.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
                exc_vals[alpha] = (m, s)

        # Self-compatibility (alpha-independent — take mean across all rows)
        sc_vals = sub["raw_self_compatibility"].dropna()
        sc_n = len(sc_vals)
        sc_m = sc_vals.mean() if sc_n > 0 else float("nan")
        sc_s = (sc_vals.std(ddof=1) / math.sqrt(sc_n)) if sc_n > 1 else float("nan")

        # alpha=0.10 metrics
        sub10 = sub[sub["alpha"].round(3) == 0.10]

        def mean_se(col: str) -> tuple[float, float]:
            if sub10.empty or col not in sub10.columns:
                return float("nan"), float("nan")
            v = sub10[col].dropna()
            n = len(v)
            m = v.mean() if n > 0 else float("nan")
            s = (v.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
            return m, s

        ap_m, ap_s = mean_se("accepted_precision")
        ar_m, ar_s = mean_se("accepted_recall")
        ur_m, ur_s = mean_se("unresolved_rate")
        f1_m, f1_s = mean_se("skeleton_f1")
        shd_m, shd_s = mean_se("full_graph_shd")

        e05 = fmt_pm(*exc_vals[0.05])
        e10 = fmt_pm(*exc_vals[0.10])
        e20 = fmt_pm(*exc_vals[0.20])
        sc = fmt_pm(sc_m, sc_s)
        ap = fmt_pm(ap_m, ap_s)
        ar = fmt_pm(ar_m, ar_s)
        ur = fmt_pm(ur_m, ur_s)
        f1 = fmt_pm(f1_m, f1_s)
        shd = fmt_pm(shd_m, shd_s)

        lines.append(
            f"\\textbf{{{label}}} & & {e05} & {e10} & {e20} & {sc} & {ap} & {ar} & {ur} & {f1} & {shd} \\\\"
        )

        if i < len(methods_present) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Synthetic experiments (30 repeats). "
        r"Empirical exceedance at three nominal levels $\alpha$, "
        r"mean self-compatibility, accepted precision/recall, unresolved rate, "
        r"skeleton F1, and SHD at $\alpha=0.10$. "
        r"Values: mean $\pm$ standard error.}"
    )
    lines.append(r"\label{tab:synthetic_main}")
    lines.append(r"\end{table}")

    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {outpath}", flush=True)


# ---------------------------------------------------------------------------
# Artifact 2b: table_baselines.tex
# ---------------------------------------------------------------------------

def make_table_baselines(df: pd.DataFrame, outpath: str) -> None:
    """
    Rows per method x estimator:
      - CSCD accepted (alpha=.1): accepted_precision/recall, SHD=--
      - Single run: single_run_precision/recall/shd
      - Stability selection: stability_precision/recall/shd
      - Uncalibrated self-comp.: uncal_precision/uncal_recall, SHD=--
    """
    ESTIMATORS = [
        ("CSCD accepted ($\\alpha{=}.1$)", "accepted_precision", "accepted_recall", None, True),
        ("Single run",                     "single_run_precision", "single_run_recall", "single_run_shd", False),
        ("Stability selection",            "stability_precision", "stability_recall", "stability_shd", False),
        ("Uncalibrated self-comp.",        "uncal_precision", "uncal_recall", None, True),
    ]

    sub10 = df[df["alpha"].round(3) == 0.10]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Estimator & Precision & Recall & SHD \\")
    lines.append(r"\midrule")

    methods_present = [m for m in METHOD_ORDER if m in df["method"].values]

    for i, method in enumerate(methods_present):
        label = METHOD_LABELS.get(method, method.upper())
        sub_m = sub10[sub10["method"] == method]
        # For alpha-independent estimators (single/stability/uncal), use all alphas
        sub_m_all = df[df["method"] == method]

        first = True
        for est_label, prec_col, rec_col, shd_col, use_alpha10 in ESTIMATORS:
            src = sub_m if use_alpha10 else sub_m_all

            def ms(col: str | None, s: pd.DataFrame) -> tuple[float, float]:
                if col is None or col not in s.columns:
                    return float("nan"), float("nan")
                v = s[col].dropna()
                n = len(v)
                m = v.mean() if n > 0 else float("nan")
                se_v = (v.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
                return m, se_v

            # For alpha-independent cols in single/stability: deduplicate by repeat
            # (they're the same regardless of alpha, so take per-repeat means first)
            if not use_alpha10 and shd_col is not None:
                # Deduplicate: same single_run / stability values repeated per alpha
                dedup = sub_m_all.drop_duplicates(subset=["repeat"])
                p_m, p_s = ms(prec_col, dedup)
                r_m, r_s = ms(rec_col, dedup)
                shd_m, shd_s = ms(shd_col, dedup)
            else:
                p_m, p_s = ms(prec_col, src)
                r_m, r_s = ms(rec_col, src)
                shd_m, shd_s = ms(shd_col, src) if shd_col else (float("nan"), float("nan"))

            prec_str = fmt_pm(p_m, p_s)
            rec_str = fmt_pm(r_m, r_s)
            shd_str = fmt_pm(shd_m, shd_s) if shd_col is not None else r"$\text{--}$"

            method_str = f"\\textbf{{{label}}}" if first else ""
            first = False
            lines.append(f"{method_str} & {est_label} & {prec_str} & {rec_str} & {shd_str} \\\\")

        if i < len(methods_present) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Comparison of CSCD ($\alpha=0.10$) to baselines. "
        r"Mean $\pm$ standard error over repeats. "
        r"SHD not applicable (--) for claim-level estimators.}"
    )
    lines.append(r"\label{tab:baselines}")
    lines.append(r"\end{table}")

    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {outpath}", flush=True)


# ---------------------------------------------------------------------------
# Artifact 2c: figure_calibration.pdf
# ---------------------------------------------------------------------------

MARKERS = ["o", "s", "^", "D", "v", "P"]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def make_figure_calibration(df: pd.DataFrame, outpath: str) -> None:
    """
    Empirical held-out exceedance (y) vs nominal alpha (x) per method,
    with y=x dashed reference, error bars = se across repeats.
    """
    methods_present = [m for m in METHOD_ORDER if m in df["method"].values]

    fig, ax = plt.subplots(figsize=(3.3, 3.0))

    for idx, method in enumerate(methods_present):
        sub = df[df["method"] == method]
        pts = []
        for alpha in sorted(sub["alpha"].unique()):
            sub_a = sub[sub["alpha"] == alpha]["cal_exceed_rate"].dropna()
            n = len(sub_a)
            if n == 0:
                continue
            m = sub_a.mean()
            s = sub_a.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
            pts.append((alpha, m, s))
        if not pts:
            continue
        alphas, means, stes = zip(*pts)
        ax.errorbar(
            alphas, means, yerr=stes,
            label=METHOD_LABELS.get(method, method.upper()),
            marker=MARKERS[idx % len(MARKERS)],
            color=COLORS[idx % len(COLORS)],
            capsize=3,
            linewidth=1.2,
            markersize=5,
        )

    xlim_max = max(max(ALPHA_LEVELS) + 0.02, 0.25)
    xs = np.linspace(0, xlim_max, 200)
    ax.plot(xs, xs, linestyle="--", color="gray", linewidth=1.0, label=r"$y = \alpha$")
    ax.set_xlim(0, xlim_max)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Nominal level $\alpha$", fontsize=9)
    ax.set_ylabel("Empirical exceedance", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout(pad=0.5)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {outpath}", flush=True)


# ---------------------------------------------------------------------------
# Artifact 2d: figure_decision.pdf
# ---------------------------------------------------------------------------

def make_figure_decision(df: pd.DataFrame, outpath: str) -> None:
    """
    Bar chart of decision_bias_{oracle,accepted,single,naive} per method at alpha=0.10.
    """
    sub10 = df[df["alpha"].round(3) == 0.10]
    methods_present = [m for m in METHOD_ORDER if m in sub10["method"].values]

    bias_cols = ["decision_bias_oracle", "decision_bias_accepted",
                 "decision_bias_single", "decision_bias_naive"]
    bias_labels = ["Oracle", "CSCD accepted", "Single run", "Naive"]
    present_cols = [c for c in bias_cols if c in sub10.columns]
    present_labels = [bias_labels[bias_cols.index(c)] for c in present_cols]

    if not present_cols or not methods_present:
        print(f"  [WARNING] No decision bias data — skipping {outpath}", flush=True)
        return

    n_methods = len(methods_present)
    n_estimators = len(present_cols)
    x = np.arange(n_methods)
    width = 0.7 / n_estimators

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    for j, (col, lbl) in enumerate(zip(present_cols, present_labels)):
        means, stes = [], []
        for method in methods_present:
            vals = sub10[sub10["method"] == method][col].dropna()
            n = len(vals)
            m = vals.mean() if n > 0 else float("nan")
            s = vals.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
            means.append(m)
            stes.append(s)
        offset = (j - (n_estimators - 1) / 2) * width
        ax.bar(
            x + offset, means, width=width * 0.9,
            label=lbl,
            color=COLORS[j % len(COLORS)],
            yerr=stes,
            capsize=2,
            error_kw={"linewidth": 0.8},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m.upper()) for m in methods_present], fontsize=9)
    ax.set_ylabel("Decision bias", fontsize=9)
    ax.set_xlabel("Method", fontsize=9)
    ax.legend(fontsize=7, framealpha=0.8)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout(pad=0.5)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {outpath}", flush=True)


# ---------------------------------------------------------------------------
# Artifact 2e: summary_digest.csv
# ---------------------------------------------------------------------------

NUMERIC_COLS = [
    "tau", "raw_self_compatibility", "n_selected_graphs", "cal_exceed_rate",
    "insample_exceed_rate", "calibration_error", "accepted_precision",
    "accepted_recall", "accepted_rate", "rejected_rate", "unresolved_rate",
    "skeleton_f1", "full_graph_shd", "single_run_precision", "single_run_recall",
    "single_run_shd", "single_run_skeleton_f1", "stability_precision",
    "stability_recall", "stability_shd", "stability_skeleton_f1",
    "uncal_precision", "uncal_recall", "uncal_unresolved_rate",
    "uncal_n_selected_graphs", "decision_bias_oracle", "decision_bias_single",
    "decision_bias_accepted", "decision_bias_naive", "decision_n_pairs",
]


def make_summary_digest(df: pd.DataFrame, outpath: str) -> None:
    """Aggregated means/ses of all numeric columns by method and alpha."""
    present_numeric = [c for c in NUMERIC_COLS if c in df.columns]
    result = _agg_with_se(df, ["method", "alpha"], present_numeric)
    result.to_csv(outpath, index=False, float_format="%.6f")
    print(f"  Wrote {outpath}", flush=True)


# ---------------------------------------------------------------------------
# Artifact 3a: table_knowledge.tex
# ---------------------------------------------------------------------------

def make_table_knowledge(combined: pd.DataFrame, outpath: str) -> None:
    """
    Rows: none, correct+hard, correct+soft, noisy+hard, noisy+soft.
    Columns (per method PC, FCI): accepted precision, accepted recall, unresolved rate
    at alpha=0.10.
    """
    sub10 = combined[combined["alpha"].round(3) == 0.10]

    methods_present = [m for m in ["pc", "fci"] if m in sub10["method"].values]

    # Build regime column
    def regime_key(row: pd.Series) -> str:
        kr = row.get("knowledge_regime", "none")
        km = row.get("knowledge_mode", "")
        if kr == "none" or pd.isna(kr):
            return "none"
        return f"{kr}+{km}"

    sub10 = sub10.copy()
    sub10["regime"] = sub10.apply(regime_key, axis=1)

    regimes_present = [r for r in REGIME_ORDER if r in sub10["regime"].values]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")

    # Build column spec
    ncol = 1 + 3 * len(methods_present)
    col_spec = "l" + "ccc" * len(methods_present)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header row 1: method spans
    header1 = "Regime"
    for method in methods_present:
        header1 += f" & \\multicolumn{{3}}{{c}}{{{METHOD_LABELS.get(method, method.upper())}}}"
    header1 += r" \\"
    lines.append(header1)

    # Cmidrule lines
    for k, method in enumerate(methods_present):
        start = 2 + 3 * k
        end = start + 2
        lines.append(f"\\cmidrule(lr){{{start}-{end}}}")

    # Header row 2: column labels
    header2 = ""
    for _ in methods_present:
        header2 += " & Prec. & Rec. & Unres."
    lines.append("Regime" + header2 + r" \\")
    lines.append(r"\midrule")

    for regime in regimes_present:
        sub_r = sub10[sub10["regime"] == regime]
        display = regime.replace("+", " + ").replace("none", "None")
        row = display
        for method in methods_present:
            sub_rm = sub_r[sub_r["method"] == method]
            for col in ["accepted_precision", "accepted_recall", "unresolved_rate"]:
                if col not in sub_rm.columns or sub_rm.empty:
                    row += r" & \text{--}"
                else:
                    vals = sub_rm[col].dropna()
                    n = len(vals)
                    m = vals.mean() if n > 0 else float("nan")
                    s = vals.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
                    row += f" & {fmt_pm(m, s)}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Effect of background knowledge at $\alpha=0.10$. "
        r"Mean $\pm$ standard error over repeats.}"
    )
    lines.append(r"\label{tab:knowledge}")
    lines.append(r"\end{table}")

    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {outpath}", flush=True)


# ---------------------------------------------------------------------------
# Artifact 3b: figure_knowledge.pdf
# ---------------------------------------------------------------------------

def make_figure_knowledge(combined: pd.DataFrame, outpath: str) -> None:
    """
    Small multi-panel: accepted precision and unresolved rate by regime/mode, per method.
    """
    sub10 = combined[combined["alpha"].round(3) == 0.10].copy()

    def regime_key(row: pd.Series) -> str:
        kr = row.get("knowledge_regime", "none")
        km = row.get("knowledge_mode", "")
        if kr == "none" or pd.isna(kr):
            return "none"
        return f"{kr}+{km}"

    sub10["regime"] = sub10.apply(regime_key, axis=1)

    methods_present = [m for m in ["pc", "fci"] if m in sub10["method"].values]
    regimes_present = [r for r in REGIME_ORDER if r in sub10["regime"].values]
    regime_labels = [r.replace("+", "\n+").replace("none", "None") for r in regimes_present]
    metrics = [("accepted_precision", "Accepted Precision"), ("unresolved_rate", "Unresolved Rate")]

    n_rows = len(metrics)
    n_cols = len(methods_present)
    if n_rows == 0 or n_cols == 0:
        print(f"  [WARNING] No data for knowledge figure — skipping {outpath}", flush=True)
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 2.5 * n_rows), squeeze=False)

    x = np.arange(len(regimes_present))

    for row_i, (metric, metric_lbl) in enumerate(metrics):
        for col_i, method in enumerate(methods_present):
            ax = axes[row_i][col_i]
            sub_m = sub10[sub10["method"] == method]
            means, stes = [], []
            for regime in regimes_present:
                vals = sub_m[sub_m["regime"] == regime][metric].dropna() if metric in sub_m.columns else pd.Series([], dtype=float)
                n = len(vals)
                m = vals.mean() if n > 0 else float("nan")
                s = vals.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
                means.append(m)
                stes.append(s)
            ax.bar(x, means, yerr=stes, capsize=3, color=COLORS[col_i % len(COLORS)],
                   error_kw={"linewidth": 0.8})
            ax.set_xticks(x)
            ax.set_xticklabels(regime_labels, fontsize=7, rotation=30, ha="right")
            ax.set_ylabel(metric_lbl, fontsize=9)
            if row_i == 0:
                ax.set_title(METHOD_LABELS.get(method, method.upper()), fontsize=9)
            ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    fig.tight_layout(pad=0.6)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {outpath}", flush=True)


# ---------------------------------------------------------------------------
# Artifact 4: table_nsens.tex
# ---------------------------------------------------------------------------

def make_table_nsens(combined: pd.DataFrame, outpath: str) -> None:
    """
    Rows: N=10, N=20, N=50 (n_calibration_resamples).
    Columns: tau, cal_exceed_rate at .05/.10/.20, accepted_precision, accepted_recall,
             unresolved_rate.
    """
    if "n_calibration_resamples" not in combined.columns and "nsens_n" not in combined.columns:
        # Try to infer n_calibration_resamples from a synthetic column added by caller
        # Caller sets it before passing
        pass

    # Group key is n_calibration_resamples (or n_cal if present)
    n_col = "n_calibration_resamples" if "n_calibration_resamples" in combined.columns else "nsens_n"
    if n_col not in combined.columns:
        print(f"  [WARNING] No N column found for nsens table — skipping {outpath}", flush=True)
        return

    n_vals = sorted(combined[n_col].dropna().unique())

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{lccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"$N$ & $\tau$ & Exc.\ $\alpha{=}.05$ & Exc.\ $\alpha{=}.10$ & Exc.\ $\alpha{=}.20$ "
        r"& Acc.\ Prec. & Acc.\ Rec. & Unres.\ Rate \\"
    )
    lines.append(r"\midrule")

    for n_val in n_vals:
        sub_n = combined[combined[n_col] == n_val]
        sub10 = sub_n[sub_n["alpha"].round(3) == 0.10]

        def ms_col(col: str, src: pd.DataFrame) -> tuple[float, float]:
            if col not in src.columns or src.empty:
                return float("nan"), float("nan")
            v = src[col].dropna()
            n = len(v)
            m = v.mean() if n > 0 else float("nan")
            s = v.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
            return m, s

        tau_m, tau_s = ms_col("tau", sub10)

        exc_strs = []
        for alpha in ALPHA_LEVELS:
            sub_a = sub_n[sub_n["alpha"].round(3) == round(alpha, 3)]
            m, s = ms_col("cal_exceed_rate", sub_a)
            exc_strs.append(fmt_pm(m, s))

        ap_m, ap_s = ms_col("accepted_precision", sub10)
        ar_m, ar_s = ms_col("accepted_recall", sub10)
        ur_m, ur_s = ms_col("unresolved_rate", sub10)

        n_display = int(n_val)
        line = (
            f"$N={n_display}$ & {fmt_pm(tau_m, tau_s)} & "
            + " & ".join(exc_strs)
            + f" & {fmt_pm(ap_m, ap_s)} & {fmt_pm(ar_m, ar_s)} & {fmt_pm(ur_m, ur_s)} \\\\"
        )
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Sensitivity to $N$ (number of calibration resamples). "
        r"PC method, values at $\alpha=0.10$ (except exceedance). "
        r"Mean $\pm$ standard error.}"
    )
    lines.append(r"\label{tab:nsens}")
    lines.append(r"\end{table}")

    with open(outpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {outpath}", flush=True)


# ---------------------------------------------------------------------------
# Artifact 5: CHECKS section (stdout)
# ---------------------------------------------------------------------------

def print_checks(df_main: pd.DataFrame) -> None:
    print("\n" + "=" * 60, flush=True)
    print("CHECKS", flush=True)
    print("=" * 60, flush=True)

    methods_present = [m for m in METHOD_ORDER if m in df_main["method"].values]

    # (i) Held-out exceedance within 2 SE of nominal alpha
    print("\n(i) Held-out exceedance within 2 SE of nominal alpha:", flush=True)
    for method in methods_present:
        sub = df_main[df_main["method"] == method]
        for alpha in ALPHA_LEVELS:
            sub_a = sub[sub["alpha"].round(3) == round(alpha, 3)]["cal_exceed_rate"].dropna()
            n = len(sub_a)
            if n == 0:
                print(f"    {method.upper()} alpha={alpha:.2f}: NO DATA", flush=True)
                continue
            m = float(sub_a.mean())
            s = float(sub_a.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
            if math.isnan(s):
                status = "UNKNOWN (n<=1)"
            else:
                status = "PASS" if abs(m - alpha) <= 2 * s else "FAIL"
            se_str = f"{s:.4f}" if not math.isnan(s) else "nan"
            print(
                f"    {method.upper()} alpha={alpha:.2f}: empirical={m:.4f}, "
                f"se={se_str}, |diff|={abs(m-alpha):.4f} → {status}",
                flush=True,
            )

    # (ii) CSCD accepted precision > single_run_precision per method
    print("\n(ii) CSCD accepted precision > single_run_precision at alpha=0.10:", flush=True)
    sub10 = df_main[df_main["alpha"].round(3) == 0.10]
    for method in methods_present:
        sub_m = sub10[sub10["method"] == method]
        if sub_m.empty:
            continue
        ap = sub_m["accepted_precision"].dropna().mean() if "accepted_precision" in sub_m.columns else float("nan")
        sp = sub_m["single_run_precision"].dropna().mean() if "single_run_precision" in sub_m.columns else float("nan")
        if math.isnan(ap) or math.isnan(sp):
            status = "NO DATA"
        else:
            status = f"PASS (CSCD={ap:.4f} > Single={sp:.4f})" if ap > sp else f"FAIL (CSCD={ap:.4f} <= Single={sp:.4f})"
        print(f"    {method.upper()}: {status}", flush=True)

    # (iii) Ordering of decision biases: accepted vs single vs naive
    print("\n(iii) Decision bias on confounded pairs (accepted < single) at alpha=0.10:", flush=True)
    bias_cols_check = ["decision_bias_oracle", "decision_bias_accepted", "decision_bias_single", "decision_bias_naive"]
    for method in methods_present:
        sub_m = sub10[sub10["method"] == method]
        if sub_m.empty:
            continue
        vals = {}
        for col in bias_cols_check:
            if col in sub_m.columns:
                v = sub_m[col].dropna()
                vals[col] = v.mean() if len(v) > 0 else float("nan")
            else:
                vals[col] = float("nan")
        b_oracle = vals.get("decision_bias_oracle", float("nan"))
        b_accept = vals.get("decision_bias_accepted", float("nan"))
        b_single = vals.get("decision_bias_single", float("nan"))
        b_naive  = vals.get("decision_bias_naive", float("nan"))
        # Primary reviewer claim: acting on accepted claims beats acting on a
        # single discovered graph.  Naive is reported as context (on
        # confounded pairs it should be the worst non-oracle strategy).
        primary_ok = (
            not math.isnan(b_accept) and not math.isnan(b_single)
            and b_accept < b_single
        )
        naive_note = ""
        if not math.isnan(b_naive) and not math.isnan(b_accept):
            naive_note = "; accepted < naive" if b_accept < b_naive else "; WARNING accepted >= naive"
        print(
            f"    {method.upper()}: oracle={b_oracle:.4f}, accepted={b_accept:.4f}, "
            f"single={b_single:.4f}, naive={b_naive:.4f} → "
            + ("PASS (accepted < single)" if primary_ok else "FAIL (accepted >= single)")
            + naive_note,
            flush=True,
        )

    print("=" * 60 + "\n", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Produce paper artifacts from experiment results.")
    parser.add_argument("--results_root", default="results",
                        help="Root directory containing experiment result subdirs.")
    parser.add_argument("--outdir", default="artifacts",
                        help="Output directory for artifacts.")
    args = parser.parse_args()

    results_root = args.results_root
    outdir = args.outdir
    ensure_dir(outdir)

    print(f"Results root : {results_root}", flush=True)
    print(f"Output dir   : {outdir}", flush=True)

    # ------------------------------------------------------------------ #
    # 2. Synthetic main
    # ------------------------------------------------------------------ #
    print("\n[Synthetic main]", flush=True)
    syn_path = os.path.join(results_root, "synthetic_main", "summary.csv")
    df_main = load_csv(syn_path)

    if df_main is not None:
        # 2a
        print("  Generating table_synthetic_main.tex …", flush=True)
        make_table_synthetic_main(df_main, os.path.join(outdir, "table_synthetic_main.tex"))

        # 2b
        print("  Generating table_baselines.tex …", flush=True)
        make_table_baselines(df_main, os.path.join(outdir, "table_baselines.tex"))

        # 2c
        print("  Generating figure_calibration.pdf …", flush=True)
        make_figure_calibration(df_main, os.path.join(outdir, "figure_calibration.pdf"))

        # 2d
        print("  Generating figure_decision.pdf …", flush=True)
        make_figure_decision(df_main, os.path.join(outdir, "figure_decision.pdf"))

        # 2e
        print("  Generating summary_digest.csv …", flush=True)
        make_summary_digest(df_main, os.path.join(outdir, "summary_digest.csv"))
    else:
        print(f"  [SKIP] synthetic_main results missing — skipping artifacts 2a-2e.", flush=True)

    # ------------------------------------------------------------------ #
    # 3. Knowledge experiment
    # ------------------------------------------------------------------ #
    print("\n[Knowledge experiment]", flush=True)
    knowledge_dirs = {
        "correct_hard": os.path.join(results_root, "knowledge_correct_hard"),
        "correct_soft": os.path.join(results_root, "knowledge_correct_soft"),
        "noisy_hard":   os.path.join(results_root, "knowledge_noisy_hard"),
        "noisy_soft":   os.path.join(results_root, "knowledge_noisy_soft"),
    }
    knowledge_regime_map = {
        "correct_hard": ("correct", "hard"),
        "correct_soft": ("correct", "soft"),
        "noisy_hard":   ("noisy",   "hard"),
        "noisy_soft":   ("noisy",   "soft"),
    }

    knowledge_dfs = []
    any_knowledge = False

    # Add reference from synthetic_main (restricted to repeat <= 9, pc/fci, regime=none)
    if df_main is not None:
        ref_main = df_main[
            (df_main["repeat"] <= 9)
            & (df_main["method"].isin(["pc", "fci"]))
        ].copy()
        if not ref_main.empty:
            knowledge_dfs.append(ref_main)
            any_knowledge = True

    for key, kdir in knowledge_dirs.items():
        kpath = os.path.join(kdir, "summary.csv")
        kdf = load_csv(kpath)
        if kdf is not None:
            any_knowledge = True
            # Ensure knowledge_regime and knowledge_mode are set correctly
            regime, mode = knowledge_regime_map[key]
            if "knowledge_regime" not in kdf.columns:
                kdf["knowledge_regime"] = regime
            if "knowledge_mode" not in kdf.columns:
                kdf["knowledge_mode"] = mode
            knowledge_dfs.append(kdf)

    if any_knowledge and knowledge_dfs:
        df_knowledge = pd.concat(knowledge_dfs, ignore_index=True)
        print("  Generating table_knowledge.tex …", flush=True)
        make_table_knowledge(df_knowledge, os.path.join(outdir, "table_knowledge.tex"))
        print("  Generating figure_knowledge.pdf …", flush=True)
        make_figure_knowledge(df_knowledge, os.path.join(outdir, "figure_knowledge.pdf"))
    else:
        print("  [SKIP] No knowledge data available — skipping artifacts 3a-3b.", flush=True)

    # ------------------------------------------------------------------ #
    # 4. N-sensitivity experiment
    # ------------------------------------------------------------------ #
    print("\n[N-sensitivity experiment]", flush=True)
    nsens_dirs = {
        10:  os.path.join(results_root, "nsens_N10"),
        50:  os.path.join(results_root, "nsens_N50"),
    }

    nsens_dfs = []
    any_nsens = False

    # Reference: synthetic_main (pc, repeat <= 9) → N=20
    if df_main is not None:
        ref_nsens = df_main[
            (df_main["repeat"] <= 9)
            & (df_main["method"] == "pc")
        ].copy()
        if not ref_nsens.empty:
            ref_nsens["n_calibration_resamples"] = 20
            nsens_dfs.append(ref_nsens)
            any_nsens = True

    for n_val, ndir in nsens_dirs.items():
        npath = os.path.join(ndir, "summary.csv")
        ndf = load_csv(npath)
        if ndf is not None:
            any_nsens = True
            ndf = ndf[ndf["method"] == "pc"].copy()
            if "n_calibration_resamples" not in ndf.columns:
                ndf["n_calibration_resamples"] = n_val
            nsens_dfs.append(ndf)

    if any_nsens and nsens_dfs:
        df_nsens = pd.concat(nsens_dfs, ignore_index=True)
        print("  Generating table_nsens.tex …", flush=True)
        make_table_nsens(df_nsens, os.path.join(outdir, "table_nsens.tex"))
    else:
        print("  [SKIP] No nsens data available — skipping artifact 4.", flush=True)

    # ------------------------------------------------------------------ #
    # 5. CHECKS
    # ------------------------------------------------------------------ #
    if df_main is not None:
        print_checks(df_main)
    else:
        print("\n[CHECKS] No synthetic_main data — cannot run checks.\n", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
