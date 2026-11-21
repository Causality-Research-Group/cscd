"""Evaluate the finite-threshold false-adjacency bound (rebuttal/bound.md) on stored records.

Usage:  python rebuttal/bound_eval.py RECORD_DIR [RECORD_DIR ...]
Writes rebuttal/out/bound_<experiment>.csv and prints a summary.
"""

import glob
import itertools
import os
import pickle
import sys

import numpy as np
import pandas as pd

from cscd.selfcompat import conformal_threshold

ALPHA = 0.1
STANDARD_TAGS = ("full", "grid", "boot", "edit")
FAILED = -1


def pair_index(names):
    return {v: i for i, v in enumerate(names)}


def separation_counts(rec):
    """c_e (all separating local graphs) and c*_e (only those correct on e) for every pair."""
    names = rec["var_names"]
    idx = pair_index(names)
    p = len(names)
    c_all = np.zeros((p, p), dtype=int)
    c_correct = np.zeros((p, p), dtype=int)
    present = np.zeros((p, p), dtype=int)  # number of subsets containing the pair
    for j, S in enumerate(rec["subsets"]):
        local = rec["local_train"][j]
        if (local == FAILED).all():
            continue
        local_adj = local != 0
        true_adj = rec["truth_proj"][j] != 0
        for a, b in itertools.combinations(range(len(S)), 2):
            u, v = sorted((idx[S[a]], idx[S[b]]))
            present[u, v] += 1
            if not local_adj[a, b]:
                c_all[u, v] += 1
                if not true_adj[a, b]:
                    c_correct[u, v] += 1
    return c_all, c_correct, present


def max_false_within_budget(costs, budget):
    """Largest number of pairs whose costs sum to at most the budget (greedy on sorted costs is exact)."""
    total, count = 0, 0
    for c in sorted(costs):
        if total + c > budget:
            break
        total += c
        count += 1
    return count


def evaluate(rec):
    names = rec["var_names"]
    p = len(names)
    iu = np.triu_indices(p, 1)
    tau = conformal_threshold(np.asarray(rec["cal_scores"]["skeleton"]), ALPHA)
    k = len(rec["subsets"][0])
    K = k * (k - 1) // 2
    m = len(rec["subsets"])
    budget = m * K * tau
    c_all, c_correct, present = separation_counts(rec)
    true_skel = rec["true_skel"] > 0
    true_nonadj = ~true_skel[iu]
    c_star = c_correct[iu][true_nonadj]
    c_min = int(c_star.min()) if c_star.size else 0

    scores = np.asarray(rec["cand_scores"]["skeleton"])
    std = np.array([t in STANDARD_TAGS for t in rec["cand_tags"]])
    admitted = np.flatnonzero(std & (scores <= tau))
    false_counts, budget_used = [], []
    for g in admitted:
        adj = rec["cand_marks"][g] != 0
        adj = adj | adj.T
        a = adj[iu]
        false_counts.append(int(np.sum(a & true_nonadj)))
        budget_used.append(int(np.sum(c_all[iu][a])))
    # Check the lemma on every admitted graph: the sum of c_e over its adjacencies fits in the budget.
    lemma_ok = all(u <= budget + 1e-9 for u in budget_used)

    return dict(
        experiment=rec.get("experiment"), method=rec["method"], repeat=rec.get("repeat"),
        m=m, k=k, tau=tau, budget=budget, n_admitted=len(admitted),
        mean_pairs_per_subset_count=float(present[iu].mean()),
        c_min_star=c_min,
        frac_nonadj_with_c_star_0=float(np.mean(c_star == 0)) if c_star.size else np.nan,
        bound_simple=(budget / c_min) if c_min >= 1 else np.inf,
        bound_tight=max_false_within_budget(c_star, budget),
        observed_max_false=max(false_counts) if false_counts else np.nan,
        observed_mean_false=float(np.mean(false_counts)) if false_counts else np.nan,
        certified_nonadjacent_pairs=int(np.sum(c_all[iu] > budget)),
        max_c=int(c_all[iu].max()),
        lemma_holds=lemma_ok,
    )


def main(dirs):
    rows = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "rec_*.pkl"))):
            with open(path, "rb") as f:
                rec = pickle.load(f)
            if rec.get("local_train") is None or rec.get("true_skel") is None:
                continue
            rows.append(evaluate(rec))
    df = pd.DataFrame(rows)
    os.makedirs("rebuttal/out", exist_ok=True)
    for exp, g in df.groupby("experiment"):
        g.to_csv(f"rebuttal/out/bound_{exp}.csv", index=False)
    cols = ["tau", "budget", "max_c", "c_min_star", "frac_nonadj_with_c_star_0", "bound_tight",
            "observed_max_false", "certified_nonadjacent_pairs"]
    print(df.groupby(["experiment", "method"])[cols].median().round(3).to_string())
    print("lemma holds in every run:", bool(df.lemma_holds.all()))


if __name__ == "__main__":
    main(sys.argv[1:])
