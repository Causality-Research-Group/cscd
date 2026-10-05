"""Figures for the second revision, from results_v3/summary/*.csv.

Usage:
    python make_figures_v3.py --paper_root ../aistats2027

Writes PDFs to <paper_root>/figures/results_rev/.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

SUMMARY = "results_v3/summary"
# Reference categorical palette (fixed order) and text inks.
C = {"pc": "#2a78d6", "ges": "#eb6834", "fci": "#1baf7a", "other": "#eda100", "violet": "#4a3aa7"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
LABEL = {"pc": "PC", "ges": "GES", "fci": "FCI"}

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.5,
    "pdf.fonttype": 42, "font.family": "serif", "mathtext.fontset": "cm",
})


def _read(name):
    path = os.path.join(SUMMARY, f"{name}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def fig_inclusion(out):
    """(a) rank of A(G*) among the N=9 calibration scores; (b) A(G*) against tau, per repeat."""
    df = _read("inclusion_per_repeat")
    if df is None:
        return
    settings = [("valid_N9", "main"), ("design_sym_N9", "symmetric"), ("hubs_v3", "hidden hubs")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.75), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.15]))
    for ax, (exp, title) in zip(axes[:3], settings):
        d = df[df.experiment == exp]
        if d.empty:
            ax.set_visible(False)
            continue
        N = int(d.N.max())
        width = 0.27
        for i, m in enumerate(("pc", "ges", "fci")):
            x = d[d.method == m]["rank"].values
            if len(x) == 0:
                continue
            h = np.bincount(x.astype(int), minlength=N + 1) / len(x)
            ax.bar(np.arange(N + 1) + (i - 1) * width, h, width=width * 0.9, color=C[m], label=LABEL[m])
        ax.axhline(1 / (N + 1), color=INK2, ls="--", lw=0.9)
        ax.set_title(title, color=INK)
        ax.set_xticks([0, 3, 6, 9])
        ax.set_xlabel(r"rank of $A(G^\ast)$")
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
        ax.axvspan(N - 0.5, N + 0.5, color=GRID, zorder=0)
        ax.text(N, 0.95, "excl.", ha="center", va="top", fontsize=6.3, color=INK2)
    axes[0].set_ylabel("fraction of repeats")
    axes[0].legend(frameon=False, loc="upper center", ncol=1, bbox_to_anchor=(0.62, 0.86))
    ax = axes[3]
    for exp, title, mk in (("valid_N9", "main", "o"), ("design_sym_N9", "symmetric", "s"), ("hubs_v3", "hidden hubs", "^")):
        d = df[(df.experiment == exp) & (df.method == "pc")]
        if exp == "valid_N9":
            d = d.iloc[:100]
        ax.scatter(d.tau, d.a_star, s=9, marker=mk, facecolor="none", edgecolor={"main": C["pc"], "symmetric": C["other"], "hidden hubs": C["violet"]}[title],
                   linewidths=0.8, label=title)
    ax.plot([0, 0.35], [0, 0.35], color=INK2, lw=0.9, ls="--")
    ax.set_xlim(0, 0.28)
    ax.set_ylim(0, 0.35)
    ax.set_xticks([0, 0.1, 0.2])
    ax.set_xlabel(r"$\tau_{0.1}$")
    ax.set_ylabel(r"$A(G^\ast)$")
    ax.set_title(r"PC, per repeat", color=INK)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.2, borderaxespad=0.1)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(os.path.join(out, "inclusion.pdf"))
    plt.close(fig)


def fig_validity(out):
    """ECDF of per-repeat training-conditional exceedance against the Beta reference."""
    df = _read("validity_per_repeat")
    if df is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.9), sharey=True)
    for ax, (exp, a) in zip(axes, (("valid_N4", 0.2), ("valid_N9", 0.1), ("valid_N19", 0.05))):
        d = df[(df.experiment == exp) & (np.isclose(df.alpha, a))]
        if d.empty:
            continue
        N = int(d.N.iloc[0])
        k = int(np.ceil((N + 1) * (1 - a)))
        t = np.linspace(0, 0.6, 301)
        ax.plot(t, beta_dist(N + 1 - k, k).cdf(t), color=INK, lw=1.2, ls="--", label=f"Beta({N + 1 - k},{k})")
        for m in ("pc", "ges", "fci"):
            x = np.sort(d[d.method == m].exceed.values)
            if len(x):
                ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=C[m], lw=1.3, label=LABEL[m])
        ax.axvline(a, color=INK2, lw=0.8, ls=":")
        ax.set_title(fr"$N={N}$, $\alpha={a}$", color=INK)
        ax.set_xlabel("per-repeat exceedance")
        ax.set_xlim(0, 0.6)
    axes[0].set_ylabel("ECDF over repeats")
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "validity.pdf"))
    plt.close(fig)


def fig_selective(out):
    df = _read("selective")
    if df is None:
        return
    dets = [("CSCD margin $A(\\widehat G)-\\tau_\\alpha$", "CSCD", C["pc"]), ("Bootstrap compat.", "Bootstrap compat.", C["ges"]),
            ("Raw $A(\\widehat G)$", r"raw $A(\widehat G)$", INK2), ("$\\kappa_G$ (Faller et al.)", r"$\kappa_G$", C["fci"]), ("Eulig violations (ratio)", "Permutation test", C["other"]),
            ("LOVO loss (Schkoda et al.)", "LOVO", C["violet"])]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.05), sharey=False)
    for ax, m in zip(axes, ("pc", "ges", "fci")):
        d = df[df.method == m]
        if d.empty:
            continue
        base = d.err_all.iloc[0]
        for det, lab, col in dets:
            x = d[d.detector == det].sort_values("abstain")
            if len(x):
                ax.plot(x.abstain, x.err_kept, marker="o", ms=3, color=col, label=lab)
        ax.axhline(base, color=INK2, ls="--", lw=0.9)
        ax.set_title(LABEL[m], color=INK)
        ax.set_xlabel("fraction of runs rejected")
    axes[0].set_ylabel("skeleton error of kept runs")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=len(l), loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.3, rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(out, "selective.pdf"))
    plt.close(fig)


def fig_knowledge(out):
    df, det = _read("knowledge"), _read("knowledge_detection")
    if df is None or det is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 1.8))
    for m in ("pc", "fci"):
        rows = []
        for r in (0, 10, 20, 30, 40, 50):
            x = df[(df.experiment == f"know_hard_rev{r}") & (df.method == m)]
            if len(x):
                rows.append((r / 100, x.gstar_mean.iloc[0], x.reject_ghat_mean.iloc[0]))
        if rows:
            rows = np.array(rows)
            axes[0].plot(rows[:, 0], rows[:, 1], marker="o", ms=3, color=C[m], label=LABEL[m])
        d = det[det.method == m].copy()
        d["r"] = d.experiment.str.replace("know_hard_rev", "").astype(float) / 100
        d = d.sort_values("r")
        axes[1].plot(d.r, d.auroc, marker="o", ms=3, color=C[m], label=LABEL[m])
    axes[0].set_ylabel(r"$G^\ast\in\mathcal{C}_\alpha$")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel(r"AUROC vs. no knowledge")
    axes[1].axhline(0.5, color=INK2, ls="--", lw=0.9)
    for ax in axes:
        ax.set_xlabel("fraction of reversed constraints")
    axes[0].legend(frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(out, "knowledge.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper_root", default="../aistats2027")
    args = ap.parse_args()
    out = os.path.join(args.paper_root, "figures", "results_rev")
    os.makedirs(out, exist_ok=True)
    for f in (fig_inclusion, fig_validity, fig_selective, fig_knowledge):
        f(out)
        print("wrote", f.__name__)


if __name__ == "__main__":
    main()
