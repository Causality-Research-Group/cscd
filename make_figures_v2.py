"""Figures for the revised results section, from results_v2/summary/*.csv.

Usage:  python make_figures_v2.py --out <paper>/figures/results_v2
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

S = "results_v2/summary"
LAB = {"pc": "PC", "ges": "GES", "fci": "FCI"}
COL = {"pc": "#1b6ca8", "ges": "#d1495b", "fci": "#2a9d8f"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})


def load(exp):
    p = os.path.join(S, f"{exp}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def row(df, method, alpha=0.1, gamma=0.9):
    d = df[(np.isclose(df.alpha, alpha)) & (np.isclose(df.gamma, gamma)) & (df.method == method)]
    d = d[d.distance == d.distance.iloc[0]] if len(d) else d
    return d.iloc[0] if len(d) else None


def fig_chunk(out):
    """Threshold, admission and G* inclusion against chunk size at n = 3200."""
    exps = [("design_N20_c40", 40), ("design_N19", 84), ("design_N9", 177)]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 1.9))
    for m in ("pc", "ges", "fci"):
        xs, ys = [], {k: ([], []) for k in ("tau", "admission", "exceedance")}
        for e, nc in exps:
            df = load(e)
            r = row(df, m) if df is not None else None
            if r is None:
                continue
            xs.append(nc)
            for k in ys:
                ys[k][0].append(r[f"{k}_mean"]); ys[k][1].append(r[f"{k}_se"])
        for ax, k, lab in zip(axes, ("tau", "admission", "exceedance"),
                              (r"threshold $\tau_{0.1}$", "fraction of candidates admitted", "held-out exceedance")):
            ax.errorbar(xs, ys[k][0], yerr=ys[k][1], marker="o", ms=3, color=COL[m], label=LAB[m], capsize=2)
            ax.set_xlabel("chunk size $n_c$ ($N$ = 20, 19, 9)")
            ax.set_ylabel(lab)
            ax.set_xscale("log")
            ax.set_xticks([40, 84, 177]); ax.set_xticklabels(["40", "84", "177"])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[2].axhline(0.1, ls="--", color="gray", lw=0.8)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "chunk_design.pdf"))
    plt.close(fig)


def fig_samplesize(out):
    ns = [200, 400, 800, 1600, 3200, 6400]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 1.9))
    for m in ("pc", "ges", "fci"):
        xs, v = [], {k: ([], []) for k in ("exceedance", "admission", "cscd_skprec")}
        for n in ns:
            df = load("design_N9" if n == 3200 else f"n{n}_N9")
            r = row(df, m) if df is not None else None
            if r is None:
                continue
            xs.append(n)
            for k in v:
                v[k][0].append(r[f"{k}_mean"]); v[k][1].append(r[f"{k}_se"])
        for ax, k, lab in zip(axes, v, ("held-out exceedance", "fraction admitted", "accepted-adjacency precision")):
            ax.errorbar(xs, v[k][0], yerr=v[k][1], marker="o", ms=3, color=COL[m], label=LAB[m], capsize=2)
            ax.set_xscale("log"); ax.set_xlabel("$n$ ($N$ = 9)"); ax.set_ylabel(lab)
    axes[0].axhline(0.1, ls="--", color="gray", lw=0.8)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sample_size.pdf"))
    plt.close(fig)


def fig_systematic(out):
    scen = [("sys_control", "ctrl"), ("sys_latent_hubs", "hubs"), ("sys_highdim", "p=50"),
            ("sys_wrongK", "wrong K"), ("sys_quadratic", "quad.")]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.0), sharey=True)
    for ax, m in zip(axes, ("pc", "ges", "fci")):
        labels, c, b, t_c, t_b = [], [], [], [], []
        for e, lab in scen:
            df = load(e)
            r = row(df, m) if df is not None else None
            if r is None:
                continue
            labels.append(lab)
            c.append(1 - r["full_admitted_mean"]); b.append(1 - r["bcq_full_admitted_mean"])
        x = np.arange(len(labels))
        ax.bar(x - 0.2, c, 0.4, color=COL[m], label="CSCD")
        ax.bar(x + 0.2, b, 0.4, color="lightgray", label="bootstrap compat.")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_title(LAB[m]); ax.set_ylim(0, 1)
    axes[0].set_ylabel(r"fraction of runs rejecting $\widehat G_{\mathrm{full}}$")
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "systematic.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_v2/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in (fig_chunk, fig_samplesize, fig_systematic):
        try:
            f(a.out)
            print("wrote", f.__name__)
        except Exception as exc:
            print("failed", f.__name__, exc)
