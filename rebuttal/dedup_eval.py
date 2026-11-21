"""Deduplication with the exact PAG key vs. the skeleton-and-collider key (answer to d2sj-W3).

Usage:  python -m rebuttal.dedup_eval RECORD_DIR
For every run, deduplicates the admitted standard-family candidates with both keys
and reports how often the partitions differ and how much claim precision changes.
"""

import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

from cscd.analysis_v3 import claim_frequencies, claims_row, dedup_weights, standard_mask, truth_claims
from cscd.pipeline_v3 import mec_key_pag
from cscd.selfcompat import conformal_threshold

ALPHA, GAMMA = 0.1, 0.9


def evaluate(rec):
    names = rec["var_names"]
    tau = conformal_threshold(np.asarray(rec["cal_scores"]["skeleton"]), ALPHA)
    scores = np.asarray(rec["cand_scores"]["skeleton"])
    admitted = np.flatnonzero(standard_mask(rec["cand_tags"]) & (scores <= tau))
    row = dict(method=rec["method"], repeat=rec["repeat"], n_admitted=len(admitted))
    if len(admitted) == 0:
        return row
    old_keys = [rec["cand_mec"][i] for i in admitted]
    new_keys = [mec_key_pag(pd.DataFrame(rec["cand_marks"][i], index=names, columns=names), names) for i in admitted]
    row["classes_old"] = len(set(old_keys))
    row["classes_pag"] = len(set(new_keys))
    marks = rec["cand_marks"][admitted]
    for kind in ("adj", "dir_def"):
        truth = truth_claims(rec["true_graph"], rec.get("true_skel"), kind)
        for label, keys in (("old", old_keys), ("pag", new_keys)):
            r = claims_row(claim_frequencies(marks, kind, dedup_weights(keys)), truth, kind, GAMMA)
            row[f"{kind}_prec_{label}"] = r["prec"]
            row[f"{kind}_nacc_{label}"] = r["n_acc"]
    return row


def main(record_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(record_dir, "rec_*.pkl"))):
        with open(path, "rb") as f:
            rec = pickle.load(f)
        if "cand_mec" in rec and rec.get("true_graph") is not None:
            rows.append(evaluate(rec))
    df = pd.DataFrame(rows)
    os.makedirs("rebuttal/out", exist_ok=True)
    df.to_csv("rebuttal/out/dedup_pag.csv", index=False)
    df["partitions_differ"] = df.classes_old != df.classes_pag
    for kind in ("adj", "dir_def"):
        df[f"{kind}_abs_change"] = (df[f"{kind}_prec_pag"] - df[f"{kind}_prec_old"]).abs()
    summary = df.groupby("method").agg(
        runs=("repeat", "size"),
        runs_partition_differs=("partitions_differ", "sum"),
        classes_old=("classes_old", "mean"), classes_pag=("classes_pag", "mean"),
        adj_prec_old=("adj_prec_old", "mean"), adj_prec_pag=("adj_prec_pag", "mean"),
        max_adj_change=("adj_abs_change", "max"),
        dir_prec_old=("dir_def_prec_old", "mean"), dir_prec_pag=("dir_def_prec_pag", "mean"),
        max_dir_change=("dir_def_abs_change", "max"))
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
