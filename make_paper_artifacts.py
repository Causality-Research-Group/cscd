"""Build the paper's tables and figures from results/<experiment>/summary.csv.

Usage:
    python make_paper_artifacts.py --results_root results --paper_root ../..

Tables are written as bare LaTeX tabulars to <paper_root>/tables/ and figures
to <paper_root>/figures/results/.  A missing experiment is skipped with a
message, so the script can be run while experiments are still in progress.
Every number in the paper's results section comes from this script.
"""

import argparse
import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALPHA = 0.10
METHOD_LABELS = {"pc": "PC", "ges": "GES", "fci": "FCI", "lingam": "LiNGAM"}
METHOD_COLORS = {"pc": "#1b6ca8", "ges": "#d1495b", "fci": "#2a9d8f", "lingam": "#8d6cab"}

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Loading and formatting
# ---------------------------------------------------------------------------

def load(results_root: str, name: str) -> Optional[pd.DataFrame]:
    path = os.path.join(results_root, name, "summary.csv")
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    return pd.read_csv(path)


def mean_se(values: pd.Series):
    values = values.dropna()
    if len(values) == 0:
        return float("nan"), float("nan")
    se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return float(values.mean()), float(se)


def pm(values: pd.Series, decimals: int = 3) -> str:
    """'$mean \\pm se$', or '--' when there is no value."""
    m, s = mean_se(values)
    if np.isnan(m):
        return "--"
    return f"${m:.{decimals}f} \\pm {s:.{decimals}f}$"


def num(values: pd.Series, decimals: int = 3) -> str:
    m, _ = mean_se(values)
    return "--" if np.isnan(m) else f"${m:.{decimals}f}$"


def at_alpha(df: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    return df[np.isclose(df["alpha"], alpha)]


def write(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {path}")


def methods_in(df: pd.DataFrame) -> List[str]:
    order = list(METHOD_LABELS)
    return [m for m in order if m in set(df["method"])]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table_synthetic(df: pd.DataFrame, path: str) -> None:
    """Calibration, set size, true-graph inclusion, and claim metrics."""
    lines = [
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Empirical exceedance} & & & \multicolumn{3}{c}{Accepted claims} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){7-9}",
        r"Method & $\alpha{=}.05$ & $\alpha{=}.10$ & $\alpha{=}.20$ & $|\mathcal{C}_\alpha|/|\mathcal{C}|$"
        r" & $G^\ast\in\mathcal{C}_\alpha$ & Precision & Recall & Unresolved \\",
        r"\midrule",
    ]
    for m in methods_in(df):
        d = df[df["method"] == m]
        exceed = [num(at_alpha(d, a)["cal_exceed_rate"]) for a in (0.05, 0.10, 0.20)]
        d10 = at_alpha(d)
        lines.append(" & ".join([
            METHOD_LABELS[m], *exceed,
            num(d10["n_selected_graphs"] / d10["n_candidates"], 2),
            num(d10["true_graph_included"], 2),
            pm(d10["accepted_precision"]), pm(d10["accepted_recall"]), pm(d10["unresolved_rate"]),
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(path, lines)


BASELINES = [
    # (label, precision column, recall column, truth-inclusion column)
    ("CSCD (ours)", "accepted_precision", "accepted_recall", "true_graph_included"),
    ("Single run", "single_run_precision", "single_run_recall", None),
    ("Stability selection", "stability_precision", "stability_recall", None),
    ("Bayesian (order MCMC)", "bayes_precision", "bayes_recall", None),
    ("Uncalibrated self-comp.", "uncal_precision", "uncal_recall", None),
    ("Bootstrap compatibility", "bootcompat_precision", "bootcompat_recall", "bootcompat_true_graph_included"),
    ("Conformal, no subsets", "nosubsets_precision", "nosubsets_recall", "nosubsets_true_graph_included"),
]


def table_baselines(df: pd.DataFrame, path: str) -> None:
    d10 = at_alpha(df)
    methods = methods_in(d10)
    lines = [
        r"\begin{tabular}{l" + "ccc" * len(methods) + "}",
        r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{METHOD_LABELS[m]}}}" for m in methods) + r" \\",
        " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(methods))),
        "Estimator & " + " & ".join(["Prec.", "Rec.", r"$G^\ast$"] * len(methods)) + r" \\",
        r"\midrule",
    ]
    for label, prec, rec, incl in BASELINES:
        if prec not in d10.columns:
            continue
        cells = [label]
        for m in methods:
            d = d10[d10["method"] == m]
            cells += [num(d[prec]), num(d[rec]), num(d[incl], 2) if incl else "--"]
        lines.append(" & ".join(cells) + r" \\")
        if label == "CSCD (ours)" or label == "Uncalibrated self-comp.":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(path, lines)


def table_subsets(results_root: str, path: str) -> None:
    rows = []
    for frac in ("0.3", "0.5", "0.7"):
        for m in ("10", "25", "50"):
            rows.append((f"subsets_k{frac}_m{m}", frac, m, "yes"))
    rows.append(("subsets_k0.3_m10_nocover", "0.3", "10", "no"))

    lines = [
        r"\begin{tabular}{ccc" + "cccc" * 2 + "}",
        r"\toprule",
        r" & & & \multicolumn{4}{c}{PC} & \multicolumn{4}{c}{FCI} \\",
        r"\cmidrule(lr){4-7} \cmidrule(lr){8-11}",
        r"$k/p$ & $m$ & Pair cover & Prec. & Rec. & Unres. & $G^\ast$ & Prec. & Rec. & Unres. & $G^\ast$ \\",
        r"\midrule",
    ]
    found = False
    for name, frac, m, cover in rows:
        df = load(results_root, name)
        if df is None:
            continue
        found = True
        d10 = at_alpha(df)
        cells = [frac, m, cover]
        for method in ("pc", "fci"):
            d = d10[d10["method"] == method]
            cells += [num(d["accepted_precision"]), num(d["accepted_recall"]),
                      num(d["unresolved_rate"]), num(d["true_graph_included"], 2)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if found:
        write(path, lines)


ROBUSTNESS = [
    ("main", "Linear-Gaussian, $p{=}20$"),
    ("distance_pag", "PAG distance"),
    ("large_p50", "$p{=}50$"),
    ("nongauss_laplace", "Laplace noise"),
    ("latent_20pct", "20\\% hidden"),
]


def table_robustness(results_root: str, path: str) -> None:
    lines = [
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"Setting & Method & Exceed. & $G^\ast\in\mathcal{C}_\alpha$ & Prec. & Rec. & Unres."
        r" & Single prec. & Stab.\ prec. \\",
        r"\midrule",
    ]
    for name, label in ROBUSTNESS:
        df = load(results_root, name)
        if df is None:
            continue
        d10 = at_alpha(df)
        methods = [m for m in methods_in(d10) if not (name == "main" and m == "ges")]
        for i, m in enumerate(methods):
            d = d10[d10["method"] == m]
            lines.append(" & ".join([
                label if i == 0 else "", METHOD_LABELS[m],
                num(d["cal_exceed_rate"]), num(d["true_graph_included"], 2),
                num(d["accepted_precision"]), num(d["accepted_recall"]), num(d["unresolved_rate"]),
                num(d["single_run_precision"]), num(d["stability_precision"]),
            ]) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write(path, lines)


KNOWLEDGE = [
    ("knowledge_none", "None"),
    ("knowledge_correct_hard", "Correct, hard"),
    ("knowledge_correct_soft", "Correct, soft"),
    ("knowledge_noisy_hard", "Noisy, hard"),
    ("knowledge_noisy_soft", "Noisy, soft"),
]


def table_knowledge(results_root: str, path: str) -> None:
    lines = [
        r"\begin{tabular}{l" + "cccc" * 2 + "}",
        r"\toprule",
        r" & \multicolumn{4}{c}{PC} & \multicolumn{4}{c}{FCI} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"Knowledge & Prec. & Rec. & Unres. & $G^\ast$ & Prec. & Rec. & Unres. & $G^\ast$ \\",
        r"\midrule",
    ]
    found = False
    for name, label in KNOWLEDGE:
        df = load(results_root, name)
        if df is None:
            continue
        found = True
        d10 = at_alpha(df)
        cells = [label]
        for m in ("pc", "fci"):
            d = d10[d10["method"] == m]
            cells += [pm(d["accepted_precision"]), pm(d["accepted_recall"]),
                      num(d["unresolved_rate"]), num(d["true_graph_included"], 2)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if found:
        write(path, lines)


REAL = [
    ("sachs", "Sachs (real)"),
    ("bnlearn_asia", "Asia"),
    ("bnlearn_child", "Child"),
    ("bnlearn_insurance", "Insurance"),
    ("bnlearn_alarm", "Alarm"),
]


def table_real(results_root: str, path: str) -> None:
    lines = [
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"Data & Method & Exceed. & $|\mathcal{C}_\alpha|/|\mathcal{C}|$ & $G^\ast\in\mathcal{C}_\alpha$"
        r" & Prec. & Rec. & Unres. & Single prec. & Stab.\ prec. \\",
        r"\midrule",
    ]
    found = False
    for name, label in REAL:
        df = load(results_root, name)
        if df is None:
            continue
        found = True
        d10 = at_alpha(df)
        for i, m in enumerate(methods_in(d10)):
            d = d10[d10["method"] == m]
            lines.append(" & ".join([
                label if i == 0 else "", METHOD_LABELS[m],
                num(d["cal_exceed_rate"]), num(d["n_selected_graphs"] / d["n_candidates"], 2),
                num(d["true_graph_included"], 2),
                num(d["accepted_precision"]), num(d["accepted_recall"]), num(d["unresolved_rate"]),
                num(d["single_run_precision"]), num(d["stability_precision"]),
            ]) + r" \\")
        lines.append(r"\midrule")
    if not found:
        return
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write(path, lines)


def table_decision(df: pd.DataFrame, path: str) -> None:
    columns = [
        ("decision_bias_oracle", "Oracle"),
        ("decision_bias_cscd_avg", "Set-avg."),
        ("decision_bias_notrejected", "Not-rej."),
        ("decision_bias_accepted", "Accepted"),
        ("decision_bias_single", "Single run"),
        ("decision_bias_naive", "Naive"),
    ]
    d10 = at_alpha(df)
    lines = [
        r"\begin{tabular}{l" + "c" * len(columns) + "}",
        r"\toprule",
        "Method & " + " & ".join(label for _, label in columns) + r" \\",
        r"\midrule",
    ]
    for m in methods_in(d10):
        d = d10[d10["method"] == m]
        lines.append(METHOD_LABELS[m] + " & " + " & ".join(pm(d[c]) for c, _ in columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(path, lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_calibration(df: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(2.6, 2.2))
    alphas = sorted(df["alpha"].unique())
    ax.plot([0, max(alphas) * 1.1], [0, max(alphas) * 1.1], "--", color="0.5", lw=0.8, label=r"$y=\alpha$")
    for m in methods_in(df):
        stats = [mean_se(df[(df["method"] == m) & np.isclose(df["alpha"], a)]["cal_exceed_rate"]) for a in alphas]
        ax.errorbar(alphas, [s[0] for s in stats], yerr=[s[1] for s in stats], marker="o", ms=3,
                    lw=1, capsize=2, color=METHOD_COLORS[m], label=METHOD_LABELS[m])
    ax.set_xlabel(r"Nominal $\alpha$")
    ax.set_ylabel("Held-out exceedance")
    ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def sweep_figure(
    results_root: str,
    path: str,
    names: Dict[int, str],
    xlabel: str,
    panels: List[tuple],
) -> None:
    """One panel per metric, x = the swept value, one line per base method."""
    frames = []
    for x, name in names.items():
        df = load(results_root, name)
        if df is not None:
            df = at_alpha(df).assign(x=x)
            df["set_fraction"] = df["n_selected_graphs"] / df["n_candidates"]
            frames.append(df)
    if not frames:
        return
    data = pd.concat(frames)
    xs = sorted(data["x"].unique())
    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 1.9))
    for ax, (col, title) in zip(axes, panels):
        for m in methods_in(data):
            stats = [mean_se(data[(data["method"] == m) & (data["x"] == x)][col]) for x in xs]
            ax.errorbar(xs, [s[0] for s in stats], yerr=[s[1] for s in stats], marker="o", ms=3,
                        lw=1, capsize=2, color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        if col == "cal_exceed_rate":
            ax.axhline(ALPHA, ls="--", color="0.5", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def figure_samplesize(results_root: str, path: str) -> None:
    sweep_figure(
        results_root, path,
        {n: f"samplesize_n{n}" for n in (800, 1600, 3200, 6400)},
        "$n$",
        [
            ("cal_exceed_rate", r"Exceedance ($\alpha=0.1$)"),
            ("true_graph_included", r"$G^\ast\in\mathcal{C}_\alpha$"),
            ("accepted_precision", "Accepted precision"),
            ("accepted_recall", "Accepted recall"),
        ],
    )


def figure_chunksize(results_root: str, path: str) -> None:
    sweep_figure(
        results_root, path,
        {c: f"chunksize_c{c}" for c in (40, 80, 120)},
        "Chunk size $n_c$",
        [
            ("tau", r"Threshold $\tau_\alpha$"),
            ("set_fraction", r"$|\mathcal{C}_\alpha|/|\mathcal{C}|$"),
            ("accepted_precision", "Accepted precision"),
            ("true_graph_included", r"$G^\ast\in\mathcal{C}_\alpha$"),
        ],
    )


# ---------------------------------------------------------------------------
# Numbers quoted in the text
# ---------------------------------------------------------------------------

def digest(results_root: str, path: str) -> None:
    """Mean and standard error of every numeric column, per experiment, method, and alpha."""
    rows = []
    for name in sorted(os.listdir(results_root)):
        if not os.path.isdir(os.path.join(results_root, name)):
            continue
        df = load(results_root, name)
        if df is None:
            continue
        numeric = df.select_dtypes(include=[np.number]).columns.drop(["repeat", "alpha"], errors="ignore")
        for (method, alpha), group in df.groupby(["method", "alpha"]):
            for col in numeric:
                m, s = mean_se(group[col])
                rows.append({"experiment": name, "method": method, "alpha": alpha,
                             "metric": col, "mean": m, "se": s, "n": int(group[col].notna().sum())})
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.4f")
    print(f"  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--paper_root", default="../..")
    args = parser.parse_args()

    tables = os.path.join(args.paper_root, "tables")
    figures = os.path.join(args.paper_root, "figures", "results")

    main_df = load(args.results_root, "main")
    if main_df is not None:
        table_synthetic(main_df, os.path.join(tables, "synthetic.tex"))
        table_baselines(main_df, os.path.join(tables, "baselines.tex"))
        table_decision(main_df, os.path.join(tables, "decision.tex"))
        figure_calibration(main_df, os.path.join(figures, "calibration.pdf"))
    table_subsets(args.results_root, os.path.join(tables, "subsets.tex"))
    table_robustness(args.results_root, os.path.join(tables, "robustness.tex"))
    table_knowledge(args.results_root, os.path.join(tables, "knowledge.tex"))
    table_real(args.results_root, os.path.join(tables, "real.tex"))
    figure_samplesize(args.results_root, os.path.join(figures, "samplesize.pdf"))
    figure_chunksize(args.results_root, os.path.join(figures, "chunksize.pdf"))
    digest(args.results_root, os.path.join(args.results_root, "digest.csv"))


if __name__ == "__main__":
    main()
