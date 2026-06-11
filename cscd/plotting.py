import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir


def make_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    agg = (
        summary
        .groupby(["dataset", "method", "alpha"], as_index=False)
        .agg({
            "tau": ["mean", "std"],
            "raw_self_compatibility": ["mean", "std"],
            "n_selected_graphs": ["mean", "std"],
            "cal_exceed_rate": ["mean", "std"],
            "calibration_error": ["mean", "std"],
            "accepted_precision": ["mean", "std"],
            "accepted_recall": ["mean", "std"],
            "skeleton_f1": ["mean", "std"],
            "unresolved_rate": ["mean", "std"],
            "accepted_rate": ["mean", "std"],
            "rejected_rate": ["mean", "std"],
            "full_graph_shd": ["mean", "std"],
        })
    )
    agg.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c
        for c in agg.columns
    ]
    return agg


def plot_score_distributions(scores: pd.DataFrame, outdir: str) -> None:
    for (dataset, method), sub in scores.groupby(["dataset", "method"]):
        plt.figure(figsize=(5.8, 4.0))
        for score_type, g in sub.groupby("score_type"):
            plt.hist(
                g["score"].values,
                bins=30,
                alpha=0.5,
                density=True,
                label=score_type,
            )
        plt.xlabel("Incompatibility score")
        plt.ylabel("Density")
        plt.title(f"Score distributions: {dataset}, {method.upper()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "plots", f"scores_{dataset}_{method}.pdf"))
        plt.close()


def plot_calibration(summary: pd.DataFrame, outdir: str) -> None:
    # Calibration plot (Proposition 1): empirical fraction of cal scores
    # exceeding τ_α (y-axis) should track the nominal level α (x-axis).
    df = (
        summary
        .groupby(["dataset", "method", "alpha"], as_index=False)["cal_exceed_rate"]
        .mean()
    )

    for dataset, sub_dataset in df.groupby("dataset"):
        plt.figure(figsize=(5.8, 4.0))
        for method, sub in sub_dataset.groupby("method"):
            sub = sub.sort_values("alpha")
            plt.plot(
                sub["alpha"],
                sub["cal_exceed_rate"],
                marker="o",
                label=method.upper(),
            )
        xs = np.linspace(0, max(df["alpha"].max(), 0.25), 100)
        plt.plot(xs, xs, linestyle="--", color="gray", label="ideal ($y=\\alpha$)")
        plt.xlabel("Nominal miscoverage level $\\alpha$")
        plt.ylabel("Empirical cal-score exceedance rate")
        plt.title(f"Calibration check: {dataset}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "plots", f"calibration_{dataset}.pdf"))
        plt.close()


def plot_claim_rates(summary: pd.DataFrame, outdir: str) -> None:
    df = (
        summary
        .groupby(["dataset", "method", "alpha"], as_index=False)
        .agg({
            "accepted_rate": "mean",
            "rejected_rate": "mean",
            "unresolved_rate": "mean",
        })
    )

    for (dataset, method), sub in df.groupby(["dataset", "method"]):
        sub = sub.sort_values("alpha")
        plt.figure(figsize=(5.8, 4.0))
        plt.plot(sub["alpha"], sub["accepted_rate"], marker="o", label="accepted")
        plt.plot(sub["alpha"], sub["rejected_rate"], marker="o", label="rejected")
        plt.plot(sub["alpha"], sub["unresolved_rate"], marker="o", label="unresolved")
        plt.xlabel("Miscoverage level $\\alpha$")
        plt.ylabel("Fraction of directed claims")
        plt.title(f"Claim status rates: {dataset}, {method.upper()}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "plots", f"claim_rates_{dataset}_{method}.pdf"))
        plt.close()


def plot_precision_recall(summary: pd.DataFrame, outdir: str) -> None:
    df = (
        summary
        .groupby(["dataset", "method", "alpha"], as_index=False)
        .agg({
            "accepted_precision": "mean",
            "accepted_recall": "mean",
        })
    )
    for dataset, sub_dataset in df.groupby("dataset"):
        plt.figure(figsize=(5.8, 4.0))
        for method, sub in sub_dataset.groupby("method"):
            plt.plot(
                sub["accepted_recall"],
                sub["accepted_precision"],
                marker="o",
                label=method.upper(),
            )
        plt.xlabel("Accepted-claim recall")
        plt.ylabel("Accepted-claim precision")
        plt.title(f"Accepted claim reliability: {dataset}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "plots", f"precision_recall_{dataset}.pdf"))
        plt.close()


def plot_raw_self_compatibility(summary: pd.DataFrame, outdir: str) -> None:
    df = (
        summary
        .groupby(["dataset", "method"], as_index=False)["raw_self_compatibility"]
        .mean()
    )
    for dataset, sub in df.groupby("dataset"):
        sub = sub.sort_values("raw_self_compatibility")
        plt.figure(figsize=(5.8, 4.0))
        plt.bar(sub["method"].str.upper(), sub["raw_self_compatibility"])
        plt.xlabel("Base discovery method")
        plt.ylabel("Mean self-compatibility loss")
        plt.title(f"Uncalibrated self-compatibility: {dataset}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "plots", f"raw_self_compatibility_{dataset}.pdf"))
        plt.close()


def plot_top_claims(claims: pd.DataFrame, outdir: str, top_k: int = 20) -> None:
    df = (
        claims
        .groupby(["dataset", "method", "alpha", "parent", "child"], as_index=False)["frequency"]
        .mean()
    )
    for (dataset, method, alpha), sub in df.groupby(["dataset", "method", "alpha"]):
        sub = sub.sort_values("frequency", ascending=False).head(top_k)
        labels = [f"{a}->{b}" for a, b in zip(sub["parent"], sub["child"])]
        plt.figure(figsize=(7.0, 5.0))
        plt.barh(range(len(sub)), sub["frequency"].values)
        plt.yticks(range(len(sub)), labels)
        plt.gca().invert_yaxis()
        plt.xlabel("Mean edge frequency in $\\mathcal{C}_\\alpha$")
        plt.title(f"Top claims: {dataset}, {method.upper()}, alpha={alpha}")
        plt.tight_layout()
        alpha_str = str(alpha).replace(".", "p")
        plt.savefig(
            os.path.join(outdir, "plots", f"top_claims_{dataset}_{method}_alpha_{alpha_str}.pdf"),
            dpi=200,
        )
        plt.close()


def save_and_plot(
    all_summary: List[pd.DataFrame],
    all_claims: List[pd.DataFrame],
    all_scores: List[pd.DataFrame],
    outdir: str,
) -> None:
    ensure_dir(outdir)
    ensure_dir(os.path.join(outdir, "plots"))

    if not all_summary:
        raise RuntimeError("No successful experiment runs.")

    summary = pd.concat(all_summary, ignore_index=True)
    claims = pd.concat(all_claims, ignore_index=True)
    scores = pd.concat(all_scores, ignore_index=True)

    summary.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    claims.to_csv(os.path.join(outdir, "claims.csv"), index=False)
    scores.to_csv(os.path.join(outdir, "scores.csv"), index=False)

    table = make_summary_table(summary)
    table.to_csv(os.path.join(outdir, "summary_table.csv"), index=False)
    with open(os.path.join(outdir, "summary_table.tex"), "w") as f:
        f.write(table.to_latex(index=False, float_format="%.3f"))

    plot_score_distributions(scores, outdir)
    plot_calibration(summary, outdir)
    plot_claim_rates(summary, outdir)
    plot_precision_recall(summary, outdir)
    plot_raw_self_compatibility(summary, outdir)
    plot_top_claims(claims, outdir)
