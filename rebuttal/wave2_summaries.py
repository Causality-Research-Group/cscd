"""Summaries of the discussion-phase experiments (answers to Kmr8-Q1, d2sj-W4, Rfqt-Q1).

Usage:  python -m rebuttal.wave2_summaries [split|latent|knowledge ...]

Reads the records in results_v3_rebuttal/ and writes one CSV per section to
rebuttal/out/ plus a Markdown report, rebuttal/out/wave2_summaries.md.

Conventions are those of the paper's v3 tables: precision is averaged over
runs with at least one accepted claim (the others abstain and are counted),
recall counts an abstaining run as 0, and rates over runs carry Wilson
intervals.
"""

import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

from cscd.analysis_v3 import claim_frequencies, claim_matrix, claims_row, point_row, standard_mask, truth_claims
from cscd.pipeline_v3 import FAILED
from cscd.selfcompat import conformal_threshold
from rebuttal.wave1_summaries import wilson

ROOT = "results_v3_rebuttal"
OUT = "rebuttal/out"
ALPHA, GAMMA = 0.1, 0.9


def records(exp):
    for path in sorted(glob.glob(os.path.join(ROOT, exp, "rec_*.pkl"))):
        with open(path, "rb") as f:
            yield pickle.load(f)


def local_errors(local: np.ndarray, truth_proj: np.ndarray):
    """Pair error rate of subset graphs and the fraction of subset graphs with an error.

    ``local`` holds endpoint marks (FAILED for a failed run), ``truth_proj``
    the skeletons of the true marginals, one per subset.
    """
    ok = local[:, 0, 0] != FAILED
    skel = local[ok] != 0
    truth = truth_proj[ok].astype(bool)
    k = truth.shape[1]
    iu = np.triu_indices(k, 1)
    wrong = skel[:, iu[0], iu[1]] != truth[:, iu[0], iu[1]]
    return float(wrong.mean()), float(wrong.any(axis=1).mean())


def base_row(rec):
    """Calibration, passing-set and local-error statistics of one run."""
    tau = conformal_threshold(np.asarray(rec["cal_scores"]["skeleton"]), ALPHA)
    row = dict(method=rec["method"], repeat=rec["repeat"], n=rec["n"], n_train=rec["n_train"],
               chunk_size=rec["chunk_size"], tau=tau, tau_zero=float(tau == 0),
               exceed=float(np.mean(np.asarray(rec["test_scores"]["skeleton"]) > tau)),
               gstar=float(rec["true_score"]["skeleton"] <= tau),
               a_gstar=rec["true_score"]["skeleton"])
    if rec.get("local_train") is not None and rec.get("truth_proj") is not None:
        row["pair_err_train"], row["any_err_train"] = local_errors(rec["local_train"], rec["truth_proj"])
        chunk = [local_errors(L, rec["truth_proj"]) for L in rec["local_cal"]]
        row["pair_err_chunk"] = float(np.mean([c[0] for c in chunk]))
        row["any_err_chunk"] = float(np.mean([c[1] for c in chunk]))
        row["all_train_correct"] = float(row["any_err_train"] == 0)
    if rec.get("subset_sufficient") is not None:
        row["frac_sufficient"] = float(np.mean(rec["subset_sufficient"]))
    if "cand_tags" in rec and len(rec["cand_tags"]) > 1:
        cs = np.asarray(rec["cand_scores"]["skeleton"])
        std = standard_mask(rec["cand_tags"])
        adm = std & (cs <= tau)
        row.update(admitted=float(adm.sum() / std.sum()), reject_ghat=float(cs[0] > tau))
        for kind in ("adj", "dir_def"):
            T = truth_claims(rec["true_graph"], rec.get("true_skel"), kind)
            r = claims_row(claim_frequencies(rec["cand_marks"][adm], kind) if adm.any() else None, T, kind, GAMMA)
            row[f"{kind}_prec"], row[f"{kind}_rec"] = r["prec"], r["rec"]
            row[f"{kind}_abstain"] = float(r["n_acc"] == 0)
            row[f"single_{kind}_prec"] = point_row(claim_matrix(rec["cand_marks"][0], kind), T, kind)["prec"]
    return row


def summarise(df: pd.DataFrame, keys):
    """Mean of every statistic per group, with Wilson intervals for rates over runs."""
    out = []
    for key, g in df.groupby(keys, sort=False):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)), runs=len(g))
        for c in g.columns:
            if c in keys or c == "repeat" or not pd.api.types.is_numeric_dtype(g[c]):
                continue
            row[c] = g[c].mean()
        for c in ("gstar", "tau_zero", "reject_ghat", "all_train_correct"):
            if c in g:
                lo, hi = wilson(g[c].sum(), len(g))
                row[f"{c}_ci"] = f"[{lo:.2f}, {hi:.2f}]"
        out.append(row)
    return pd.DataFrame(out)


def sec_split():
    rows = []
    for f in (30, 40, 50, 60, 70, 80):
        for rec in records(f"split_f{f}"):
            rows.append(dict(train_frac=f / 100, **base_row(rec)))
    df = pd.DataFrame(rows)
    return summarise(df, ["method", "train_frac"])


def sec_latent():
    rows = []
    for n in (800, 1600, 3200, 6400, 12800):
        for rec in records(f"latent_n{n}"):
            rows.append(base_row(rec))
    df = pd.DataFrame(rows)
    return summarise(df, ["method", "n"])


def sec_knowledge():
    rows = []
    for exp, label in (("know_hard_rev0", "correct, subsets and full runs"),
                       ("know_hard_rev0_nolocal", "correct, full runs only"),
                       ("know_hard_rev30", "30% reversed, subsets and full runs"),
                       ("know_hard_rev30_nolocal", "30% reversed, full runs only")):
        for rec in records(exp):
            rows.append(dict(knowledge=label, **base_row(rec)))
    df = pd.DataFrame(rows)
    return summarise(df, ["method", "knowledge"])


SECTIONS = dict(split=sec_split, latent=sec_latent, knowledge=sec_knowledge)


def main():
    names = sys.argv[1:] or list(SECTIONS)
    os.makedirs(OUT, exist_ok=True)
    report = []
    pd.set_option("display.width", 250)
    for name in names:
        df = SECTIONS[name]()
        df.to_csv(os.path.join(OUT, f"wave2_{name}.csv"), index=False)
        text = df.round(3).to_string(index=False)
        print(f"== {name}\n{text}\n")
        report.append(f"## {name}\n\n```\n{text}\n```\n")
    with open(os.path.join(OUT, "wave2_summaries.md"), "w") as f:
        f.write("# Discussion-phase experiments\n\n" + "\n".join(report))


if __name__ == "__main__":
    main()
