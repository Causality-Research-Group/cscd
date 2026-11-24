"""Redefined "Conformal, no subsets" baseline (AI review 3.1).

The paper's version compared G-hat = A(D_train) with itself (S = {V}), so its
score was 0 and its margin -tau.  Two non-degenerate versions are computed
here, on the same data, split and G-hat as the stored records of
results_v3/<exp> (verified per record: the regenerated G-hat must equal the
stored cands[0]):

  full-data  reference R = G-hat (full-variable run on D_train); calibration
             scores d(A(chunk_b), R) on the N = 9 calibration chunks; the
             tested graph is the full-data run A(D) on D = D_train + D_cal
             (the reading of AI-review item 3.1).  Not strictly conformal,
             since A(D) uses the calibration rows.
  split      D_train is halved; R = A(D_train,a); the tested graph is
             A(D_train,b); calibration scores d(A(chunk_b), R).  Every graph
             is a function of D_train or of its own chunk, so the threshold is
             valid; the tested graph uses n_train/2 rows.

Both flag when the tested graph's skeleton distance to R exceeds the
conformal threshold at alpha = 0.1.  The raw distances are also stored for
AUROC against the control scenario.

Usage:  .venv/bin/python -m rebuttal.wave3_nosub [--workers 9]
Output: rebuttal/out/wave3/nosub_per_run.csv
"""

import argparse
import glob
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import numpy as np
import pandas as pd

from cscd.config import ExperimentConfig
from cscd.graph import discover_marks
from cscd.mixed_graph import NO_EDGE
from cscd.scenarios import make_scenario
from cscd.selfcompat import conformal_threshold, project
from cscd.utils import make_domain_knowledge

BASE_CFG = dict(n=3200, p=20, n_subsets=25, subset_frac=0.5, n_candidate_graphs=100,
                stability_n_bootstraps=50, random_seed=2027)
EXPS = {
    "main_v3": dict(cfg={}, data={}),
    "hubs_v3": dict(cfg=dict(p=26, expected_degree=3.0), data=dict(n_latent=6, latent_mode="hubs")),
    "highdim_v3": dict(cfg=dict(p=50, n=1000, n_subsets=50, subset_frac=0.3, n_candidate_graphs=40), data={}),
    "lowdim_n1000_v3": dict(cfg=dict(p=20, n=1000, n_subsets=50, subset_frac=0.75, n_candidate_graphs=40), data={}),
    "quadratic_v3": dict(cfg={}, data=dict(scenario="quadratic", quadratic_frac=0.5)),
}
N_CHUNKS = 9
ALPHA = 0.1


def _offset(method: str) -> int:
    return sum(ord(c) * (i + 1) for i, c in enumerate(method)) % 100_000


def _skel(marks: pd.DataFrame, names) -> np.ndarray:
    iu = np.triu_indices(len(names), 1)
    return (project(marks, names, "skeleton").loc[names, names].values != NO_EDGE)[iu]


def one(path: str) -> dict:
    rec = pickle.load(open(path, "rb"))
    exp, r, method = rec["experiment"], rec["repeat"], rec["method"]
    spec = EXPS[exp]
    cfg = replace(ExperimentConfig(**BASE_CFG), **spec["cfg"])
    d = spec["data"]
    seed = cfg.random_seed + r
    rng = np.random.default_rng(seed)
    sc = make_scenario(d.get("scenario", "linear"), cfg.p, cfg.n, 2000, rng, expected_degree=cfg.expected_degree,
                       n_latent=d.get("n_latent", 0), latent_mode=d.get("latent_mode", "random"),
                       quadratic_frac=d.get("quadratic_frac", 0.0))
    data = sc.data
    names = list(data.columns)
    K = make_domain_knowledge(names)
    tr = np.asarray(rec["_D_tr_index"])
    cal_idx = np.setdiff1d(np.arange(len(data)), tr)
    D_tr = data.iloc[tr].reset_index(drop=True)
    D_cal = data.iloc[cal_idx].reset_index(drop=True)
    lrng = np.random.default_rng(seed * 7 + _offset(method))

    def run(D):
        return discover_marks(D, method, K, cfg, random_state=int(lrng.integers(1_000_000)))

    G_hat = run(D_tr)
    stored = rec["cand_marks"][0]
    same = bool(np.array_equal(G_hat.loc[names, names].values.astype(np.int8), stored))
    s_hat = _skel(G_hat, names)

    size = rec["chunk_size"]
    perm = lrng.permutation(len(D_cal))
    chunks = [D_cal.iloc[perm[b * size:(b + 1) * size]].reset_index(drop=True) for b in range(N_CHUNKS)]
    s_chunks = [_skel(run(C), names) for C in chunks]

    # Full-data version.
    s_full = _skel(run(data), names)
    cal_full = np.array([np.mean(s != s_hat) for s in s_chunks])
    tau_full = conformal_threshold(cal_full, ALPHA)
    score_full = float(np.mean(s_full != s_hat))

    # Split version.
    h = lrng.permutation(len(D_tr))
    half = len(D_tr) // 2
    s_a = _skel(run(D_tr.iloc[h[:half]].reset_index(drop=True)), names)
    s_b = _skel(run(D_tr.iloc[h[half:]].reset_index(drop=True)), names)
    cal_split = np.array([np.mean(s != s_a) for s in s_chunks])
    tau_split = conformal_threshold(cal_split, ALPHA)
    score_split = float(np.mean(s_b != s_a))

    # CSCD native flag of the same record, for comparison.
    cal = np.asarray(rec["cal_scores"]["skeleton"])
    tau = conformal_threshold(cal, ALPHA)
    a_hat = float(rec["cand_scores"]["skeleton"][0])
    true_skel = rec["true_skel"].astype(bool)[np.triu_indices(len(names), 1)]
    return dict(experiment=exp, repeat=r, method=method, ghat_reproduced=same,
                skel_err_ghat=float(np.mean(s_hat != true_skel)),
                nosub_full_score=score_full, nosub_full_tau=tau_full, nosub_full_flag=float(score_full > tau_full),
                nosub_split_score=score_split, nosub_split_tau=tau_split, nosub_split_flag=float(score_split > tau_split),
                cscd_margin=a_hat - tau, cscd_flag=float(a_hat > tau))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--out", default="rebuttal/out/wave3/nosub_per_run.csv")
    args = ap.parse_args()
    paths = sorted(p for e in EXPS for p in glob.glob(f"results_v3/{e}/rec_*.pkl"))
    done = set()
    rows = []
    if os.path.exists(args.out):
        old = pd.read_csv(args.out)
        rows = old.to_dict("records")
        done = {(x["experiment"], int(x["repeat"]), x["method"]) for x in rows}
    todo = []
    for p in paths:
        b = os.path.basename(p)[4:-4].split("_")
        key = (p.split("/")[1], int(b[0]), b[1])
        if key not in done:
            todo.append(p)
    print(f"{len(todo)} records", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, p): p for p in todo}
        for i, f in enumerate(as_completed(futs)):
            try:
                rows.append(f.result())
            except Exception as exc:
                print("FAIL", futs[f], exc, flush=True)
            if i % 20 == 0:
                pd.DataFrame(rows).to_csv(args.out, index=False)
                print(i, flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
