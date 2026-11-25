"""Rebuttal, wave 1: analyses that need only the committed summaries.

Reads results_v3/summary/*.csv (no raw records) and writes
rebuttal/out/wave1_summaries.md with:

  A. Exceedance with intervals that resample whole runs (cluster bootstrap),
     next to the pooled Clopper-Pearson interval of the paper (d2sj-W2).
  B. Wilson intervals over runs for every "x% of runs" statistic:
     inclusion of G*, native flag rates (d2sj-W2).
  C. Claim-level false discovery proportion per run (d2sj-W1).
  D. Subsets x calibration: what each ingredient contributes (Rfqt-Q2, xFQE-W2).
  E. Run-level gating with the native flag across a mixed pool of runs (Kmr8-W2).

Usage:  python rebuttal/wave1_summaries.py
"""

import os

import numpy as np
import pandas as pd

SUMMARY = "results_v3/summary"
OUT_DIR = "rebuttal/out"
N_BOOT = 10_000
RNG = np.random.default_rng(2027)

CSCD_MARGIN = "CSCD margin $A(\\widehat G)-\\tau_\\alpha$"
BOOT_OWN = "Bootstrap compat. (own quantile)"
UNCAL = "Uncal. threshold"
NOSUB = "Conformal, no subsets"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(SUMMARY, f"{name}.csv"))


def cluster_bootstrap_mean(values: np.ndarray) -> tuple:
    """95% interval of the mean, resampling whole runs with replacement."""
    values = np.asarray(values, dtype=float)
    idx = RNG.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def wilson(successes: float, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a proportion over independent runs."""
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def fmt_ci(lo: float, hi: float, digits: int = 3) -> str:
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def to_markdown(df: pd.DataFrame) -> str:
    """Plain markdown table (avoids the optional 'tabulate' dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# A. Exceedance with cluster-bootstrap intervals
# ---------------------------------------------------------------------------

def section_a() -> str:
    per_run = read("validity_per_repeat")
    pooled = read("validity").set_index(["experiment", "method", "alpha"])
    rows = []
    for (exp, alpha, method), g in per_run.groupby(["experiment", "alpha", "method"]):
        exc = g.exceed.values
        lo, hi = cluster_bootstrap_mean(exc)
        cp = pooled.loc[(exp, method, alpha)]
        rows.append({
            "experiment": exp, "N": int(g.N.iloc[0]), "alpha": alpha, "method": method,
            "runs": len(g),
            "exceedance": f"{exc.mean():.3f}",
            "pooled CP 95% (paper)": fmt_ci(cp.ci_lo, cp.ci_hi),
            "cluster bootstrap 95%": fmt_ci(lo, hi),
            "width ratio": f"{(hi - lo) / (cp.ci_hi - cp.ci_lo):.2f}",
            "upper end <= alpha": "yes" if hi <= alpha + 1e-12 else "no",
            "per-run 2.5-97.5%": fmt_ci(np.quantile(exc, 0.025), np.quantile(exc, 0.975)),
            "P(run > 2 alpha)": f"{np.mean(exc > 2 * alpha):.3f}",
        })
    df = pd.DataFrame(rows)
    text = ["## A. Exceedance: intervals that resample whole runs (d2sj-W2)", "",
            "Each run has 40 held-out chunks that share the run's local graphs and threshold. "
            "The cluster bootstrap resamples whole runs (10,000 resamples), so it accounts for that dependence. "
            "'Width ratio' is the cluster interval's width divided by the pooled Clopper-Pearson width.", "",
            to_markdown(df), ""]
    return "\n".join(text)


# ---------------------------------------------------------------------------
# B. Wilson intervals over runs
# ---------------------------------------------------------------------------

def section_b() -> str:
    inc = read("inclusion_per_repeat")
    rows = []
    for exp in ["valid_N9", "main_v3", "design_sym_N9", "hubs_v3", "quadratic_v3", "n200_N9"]:
        for method, g in inc[inc.experiment == exp].groupby("method"):
            k, n = g.included.sum(), len(g)
            lo, hi = wilson(k, n)
            rows.append({"statistic": "G* retained", "experiment": exp, "method": method,
                         "runs": n, "rate": f"{k / n:.2f}", "Wilson 95%": fmt_ci(lo, hi, 2)})
    pool = read("selective_pool")
    for exp in ["main_v3", "hubs_v3", "highdim_v3", "lowdim_n1000_v3", "quadratic_v3"]:
        for method, g in pool[pool.experiment == exp].groupby("method"):
            flags = (g[CSCD_MARGIN] > 0).values
            lo, hi = wilson(flags.sum(), len(flags))
            rows.append({"statistic": "native flag on G-hat", "experiment": exp, "method": method,
                         "runs": len(flags), "rate": f"{flags.mean():.2f}", "Wilson 95%": fmt_ci(lo, hi, 2)})
    df = pd.DataFrame(rows)
    return "\n".join(["## B. Rates over runs with Wilson intervals (d2sj-W2)", "",
                      "Runs are independent (one seed each), so a binomial interval over runs is valid for these statistics.", "",
                      to_markdown(df), ""])


# ---------------------------------------------------------------------------
# C. Claim-level false discovery proportion per run
# ---------------------------------------------------------------------------

ESTIMATORS = ["CSCD", "CSCD, MEC-dedup.", "Pool, no filter", "Bootstrap freq., $D$",
              "Stability sel., $D$", "Order MCMC, $D$", "Single run, $D$"]


def section_c() -> str:
    claims = read("claims_per_repeat")
    claims = claims[claims.experiment == "main_v3"]
    rows = []
    for kind in ["adj", "dir_def", "anc"]:
        for method in ["pc", "ges", "fci"]:
            for est in ESTIMATORS:
                g = claims[(claims.kind == kind) & (claims.method == method) & (claims.estimator == est)]
                if g.empty:
                    continue
                has_claims = g.prec.notna() & (g.n_acc.fillna(0) > 0)
                fdp = 1.0 - g.prec[has_claims].values
                n_runs, n_with = len(g), int(has_claims.sum())
                if n_with == 0:
                    continue
                any_false = (fdp > 1e-12)
                lo_m, hi_m = cluster_bootstrap_mean(fdp)
                lo_p, hi_p = wilson(any_false.sum(), n_with)
                rows.append({
                    "claims": kind, "method": method, "estimator": est,
                    "runs": n_runs, "runs w/o claims": n_runs - n_with,
                    "mean accepted": f"{g.n_acc[has_claims].mean():.1f}",
                    "mean FDP [95%]": f"{fdp.mean():.3f} {fmt_ci(lo_m, hi_m)}",
                    "P(FDP>0) [95%]": f"{any_false.mean():.2f} {fmt_ci(lo_p, hi_p, 2)}",
                    "P(FDP>0.1)": f"{np.mean(fdp > 0.1):.2f}",
                    "max FDP": f"{fdp.max():.3f}",
                })
    df = pd.DataFrame(rows)
    return "\n".join(["## C. Claim-level error per run, main setting (d2sj-W1)", "",
                      "FDP = fraction of a run's accepted claims that are false. Runs with no accepted claim are counted separately "
                      "(they abstain) and excluded from the FDP statistics. Intervals: cluster bootstrap over runs for the mean, "
                      "Wilson over runs for P(FDP>0). 30 runs per method.", "",
                      to_markdown(df), ""])


# ---------------------------------------------------------------------------
# D. Subsets x calibration (2 x 2)
# ---------------------------------------------------------------------------

TWO_BY_TWO = [
    ("subsets + calibration (CSCD margin)", CSCD_MARGIN, True),
    ("subsets, uncalibrated threshold (mean pairwise score)", UNCAL, True),
    ("subsets, no threshold: raw A(G-hat)", "Raw $A(\\widehat G)$", False),
    ("subsets, no threshold: kappa_G", "$\\kappa_G$ (Faller et al.)", False),
    ("no subsets, bootstrap own quantile", BOOT_OWN, True),
    ("no subsets, bootstrap score", "Bootstrap compat.", False),
    ("no subsets, conformal (DEGENERATE, see AI-3.1)", NOSUB, True),
]


def section_d() -> str:
    det = read("detection")
    rows = []
    for scen in ["Hidden hubs", "$p{=}50$, $n{=}1000$", "Quadratic (shared)"]:
        for method in ["pc", "ges", "fci"]:
            d = det[(det.scenario == scen) & (det.method == method)].set_index("detector")
            if d.empty:
                continue
            for label, key, native in TWO_BY_TWO:
                if key not in d.index:
                    continue
                r = d.loc[key]
                rows.append({
                    "scenario": scen, "method": method, "ingredients": label,
                    "AUROC": f"{r.auroc:.2f}", "power @10% FPR": f"{r.power:.2f}",
                    "native flag: scenario / control": (f"{r.native_flag:.2f} / {r.native_flag_ctrl:.2f}" if native
                                                        else "needs a control sample"),
                })
    df = pd.DataFrame(rows)
    return "\n".join(["## D. What subsets and calibration each contribute (Rfqt-Q2, xFQE-W2)", "",
                      "Power at a matched 10% false-flag rate needs a control sample to set the threshold, which a practitioner "
                      "does not have. The native flag uses only the method's own rule. A good detector has a high native flag in "
                      "the scenario and a native flag near alpha = 0.1 or below in the control.", "",
                      to_markdown(df), ""])


# ---------------------------------------------------------------------------
# E. Run-level gating across a mixed pool of runs
# ---------------------------------------------------------------------------

def section_e() -> str:
    pool = read("selective_pool")
    pool = pool[pool.experiment.isin(["main_v3", "hubs_v3", "highdim_v3", "lowdim_n1000_v3", "quadratic_v3"])]
    rows = []
    for method, g in pool.groupby("method"):
        for label, key in [("CSCD native flag", CSCD_MARGIN), ("Uncal. threshold", UNCAL),
                           ("Bootstrap own quantile", BOOT_OWN)]:
            x = g.dropna(subset=[key])
            flag = x[key].values > 0
            err = x.skel_err_rate.values
            rows.append({
                "method": method, "rule": label, "runs": len(x),
                "flagged": f"{flag.mean():.2f}",
                "skel. error, all runs": f"{err.mean():.3f}",
                "kept runs": f"{err[~flag].mean():.3f}" if (~flag).any() else "-",
                "flagged runs": f"{err[flag].mean():.3f}" if flag.any() else "-",
                "error reduction": f"{1 - err[~flag].mean() / err.mean():.0%}" if (~flag).any() else "-",
            })
    df = pd.DataFrame(rows)
    by_exp = (pool.assign(flag=pool[CSCD_MARGIN] > 0)
              .groupby(["method", "experiment"]).agg(runs=("flag", "size"), flagged=("flag", "mean"),
                                                     err=("skel_err_rate", "mean")).round(3).reset_index())
    return "\n".join(["## E. Gating runs with each rule's own threshold (Kmr8-W2, Rfqt-W3)", "",
                      "Pool of runs from the main setting, hidden hubs, p=50, the p=20/n=1000 control and the quadratic scenario. "
                      "Each rule flags with its own threshold, no control sample. 'Skel. error' is the skeleton error rate of G-hat.", "",
                      to_markdown(df), "", "Composition of the pool (CSCD flag rate and error by experiment):", "",
                      to_markdown(by_exp), ""])


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    parts = ["# Rebuttal wave 1: analyses from committed summaries", "",
             "Generated by rebuttal/wave1_summaries.py from results_v3/summary.", "",
             section_a(), section_b(), section_c(), section_d(), section_e()]
    path = os.path.join(OUT_DIR, "wave1_summaries.md")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    print("wrote", path)


if __name__ == "__main__":
    main()
