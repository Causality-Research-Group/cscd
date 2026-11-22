"""Corrected "conformal, no subsets" baseline (answer to AI-3.1).

Usage:  python -m rebuttal.nosub_baseline [--workers 8]

In the paper, this baseline used S = {V}: the training reference was the
full-variable training run itself, so the tested graph was compared with
itself, its score was 0, and it could never flag (the AI review's point).

Corrected version, with everything else as in CSCD:
* reference:    R = A(D_train), the full-variable training run;
* calibration:  s_i = d(A(C_i), R) for the N = 9 calibration chunks C_i,
                and tau = their conformal quantile at alpha = 0.1;
* tested graph: the full-data run A(D), D = D_train + D_cal, which is not
                identical to R; its score is d(A(D), R).
The run is flagged when d(A(D), R) > tau.  A(D) uses the calibration data,
so the finite-sample guarantee does not apply; it is a baseline only.

The data are regenerated with experiments_v3.make_inputs, so they are those
of the paper's detection experiments.  Writes rebuttal/out/nosub_baseline.csv
(one row per run) and nosub_detection.csv (one row per scenario and method).
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments_v3 import _offset, make_inputs  # noqa: E402
from cscd.analysis_v3 import auroc  # noqa: E402
from cscd.graph import discover_marks  # noqa: E402
from cscd.pipeline import _chunks  # noqa: E402
from cscd.pipeline_v3 import _skel  # noqa: E402
from cscd.selfcompat import conformal_threshold  # noqa: E402

ALPHA = 0.1
# Scenario, scenario experiment, control experiment, methods (as in analyze_v3.DET_PAIRS).
PAIRS = [
    ("Hidden hubs", "hubs_v3", "main_v3", ["pc", "ges", "fci"]),
    ("p=50, n=1000", "highdim_v3", "lowdim_n1000_v3", ["pc", "ges", "fci"]),
    ("Wrong knowledge", "know_hard_rev50", "know_none", ["pc", "fci"]),
    ("Quadratic (shared)", "quadratic_v3", "main_v3", ["pc", "fci"]),
]
REPEATS = {"hubs_v3": 30, "main_v3": 30, "highdim_v3": 20, "lowdim_n1000_v3": 20,
           "know_hard_rev50": 20, "know_none": 20, "quadratic_v3": 30}


def one_run(task):
    exp, repeat, method = task
    x = make_inputs(exp, repeat)
    cfg, opts, data, knowledge = x["cfg"], x["opts"], x["data"], x["knowledge"]
    names = list(data.columns)
    rng = np.random.default_rng(x["seed"] * 1000 + _offset(method))
    # Same split rule as fit_repeat_v3.
    perm = rng.permutation(len(data))
    n_tr = int(round(opts.train_frac * len(data)))
    D_tr = data.iloc[perm[:n_tr]].reset_index(drop=True)
    D_cal = data.iloc[perm[n_tr:]].reset_index(drop=True)
    n_split = opts.n_cal_chunks + (opts.n_test_chunks if x["test_data"] is None else 0)
    chunks = _chunks(D_cal, n_split, max(1, len(D_cal) // n_split), rng)[:opts.n_cal_chunks]

    def skeleton(D):
        G = discover_marks(D, method, knowledge, cfg, random_state=int(rng.integers(1_000_000)))
        return _skel(G, names)[np.triu_indices(len(names), 1)]

    ref = skeleton(D_tr)
    cal = np.array([np.mean(skeleton(C) != ref) for C in chunks])
    tau = conformal_threshold(cal, ALPHA)
    s_full = float(np.mean(skeleton(data) != ref))
    return dict(experiment=exp, repeat=repeat, method=method, tau=tau, score=s_full,
                margin=s_full - tau, flag=float(s_full > tau))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    exps = {}
    for _, s, c, methods in PAIRS:
        for e in (s, c):
            exps.setdefault(e, set()).update(methods)
    tasks = [(e, r, m) for e, ms in exps.items() for r in range(REPEATS[e]) for m in sorted(ms)]
    print(f"{len(tasks)} runs", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(one_run, tasks))
    df = pd.DataFrame(rows)
    os.makedirs("rebuttal/out", exist_ok=True)
    df.to_csv("rebuttal/out/nosub_baseline.csv", index=False)

    out = []
    for scen, s, c, methods in PAIRS:
        for m in methods:
            S = df[(df.experiment == s) & (df.method == m)]
            C = df[(df.experiment == c) & (df.method == m)]
            out.append(dict(scenario=scen, method=m, flag_scenario=S.flag.mean(), flag_control=C.flag.mean(),
                            auroc=auroc(S.margin.values, C.margin.values),
                            score_median=S.score.median(), tau_median=S.tau.median()))
    summary = pd.DataFrame(out)
    summary.to_csv("rebuttal/out/nosub_detection.csv", index=False)
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
