"""Rebuttal experiments, wave 2 (branch rebuttal/experiments-chris).

Experiments that the draft responses promise "during the discussion", plus
two that close open review points.  Records go to results_v3_chris/<exp>/
(gitignored); summaries are written by rebuttal/analyze_chris.py.

  trainfrac_<f>        Kmr8-Q1   sweep of the training fraction, N = 9.
  nscale_<lin|hubs>_n  d2sj-W4   sample-size scaling of local error, P(tau=0)
                                 and inclusion, PC vs FCI, with and without
                                 hidden confounders (score only).
  kscope_<scope>_rev<r> Rfqt-Q1  where hard knowledge is used: candidates only,
                                 local runs only (paper: everywhere), with
                                 correct and 30%-reversed knowledge.
  hubdose_h<h>         Rfqt-W3/Kmr8-W2  dose-response of the native flag in
                                 the number of hidden hubs.

Seeds follow experiments_v3.py: repeat r of a preset draws the same graph and
data as repeat r of every other preset with the same data settings, and the
method-specific split seed is unchanged.

Usage:
    .venv/bin/python -m rebuttal.exp_chris --list
    .venv/bin/python -m rebuttal.exp_chris trainfrac_50 ... [--workers 9] [--repeats R]
"""

import argparse
import logging
import os
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from typing import Dict

import numpy as np

from cscd.config import ExperimentConfig
from cscd.pipeline_v3 import OptionsV3, fit_repeat_v3
from cscd.scenarios import make_scenario
from cscd.utils import corrupt_knowledge, knowledge_from_true_graph, make_domain_knowledge

ROOT = os.environ.get("CSCD_CHRIS_ROOT", "results_v3_chris")

BASE_CFG = dict(n=3200, p=20, n_subsets=25, subset_frac=0.5, n_candidate_graphs=100,
                stability_n_bootstraps=50, random_seed=2027)
PCGESFCI = ["pc", "ges", "fci"]
LIGHT = dict(run_ablations=False)
HUBS = dict(n_latent=6, latent_mode="hubs")
HUBS_CFG = dict(p=26, expected_degree=3.0)


def _exp(methods, repeats, cfg=None, opts=None, data=None, **extra) -> Dict:
    return dict(methods=methods, repeats=repeats, cfg=cfg or {}, opts=opts or {}, data=data or {}, **extra)


EXPERIMENTS: Dict[str, Dict] = {}

# Kmr8-Q1: training fraction 0.3 ... 0.8 at N = 9 (0.5 is the paper's design).
for f in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    # Score only (the machine was shared during the rebuttal): inclusion of G*,
    # exceedance, tau, local errors and the flag on G-hat; no candidate family.
    EXPERIMENTS[f"trainfrac_{int(100 * f)}"] = _exp(PCGESFCI, 30, opts=dict(train_frac=f, n_test_chunks=20,
                                                                            score_only=True))

# d2sj-W4: sample-size scaling, score only (local errors, tau, A(G*), A(G-hat)).
for n in (400, 800, 1600, 3200, 6400, 12800, 25600):
    EXPERIMENTS[f"nscale_lin_n{n}"] = _exp(["pc", "fci"], 30, cfg=dict(n=n),
                                            opts=dict(score_only=True, n_test_chunks=10))
    EXPERIMENTS[f"nscale_hubs_n{n}"] = _exp(["pc", "fci"], 30, cfg=dict(n=n, **HUBS_CFG), data=HUBS,
                                             opts=dict(score_only=True, n_test_chunks=10))

# d2sj-W4: the same with a significance level that shrinks with n,
# alpha_n = 0.05 * 1600 / n, as consistency of PC/FCI requires (a fixed alpha
# leaves a floor of spurious adjacencies).
for n in (6400, 25600, 102400):
    a = 0.05 * 1600 / n
    EXPERIMENTS[f"nscalea_lin_n{n}"] = _exp(["pc", "fci"], 30, cfg=dict(n=n, pc_alpha=a, fci_alpha=a),
                                             opts=dict(score_only=True, n_test_chunks=10))
    EXPERIMENTS[f"nscalea_hubs_n{n}"] = _exp(["pc", "fci"], 30, cfg=dict(n=n, pc_alpha=a, fci_alpha=a, **HUBS_CFG),
                                              data=HUBS, opts=dict(score_only=True, n_test_chunks=10))

# d2sj-W4 / Thm 2 / Prop 2: the same with the standard candidate family, to
# count false adjacencies of admitted graphs as n grows.
for n in (1600, 6400, 25600):
    EXPERIMENTS[f"nscand_hubs_n{n}"] = _exp(["pc", "fci"], 15, cfg=dict(n=n, **HUBS_CFG), data=HUBS,
                                             opts=dict(n_test_chunks=10, **LIGHT))
    EXPERIMENTS[f"nscand_lin_n{n}"] = _exp(["pc", "fci"], 15, cfg=dict(n=n), opts=dict(n_test_chunks=10, **LIGHT))

# Rfqt-Q1: scope of hard knowledge ("all" is the paper's design; its first 20
# repeats reproduce results_v3/know_hard_rev0 and know_hard_rev30), and the
# no-knowledge control (= results_v3/know_none).
EXPERIMENTS["kscope_none"] = _exp(["pc", "fci"], 30, opts=dict(run_ablations=True))
for scope in ("all", "candidates", "local"):
    for r in (0.0, 0.3):
        EXPERIMENTS[f"kscope_{scope}_rev{int(100 * r)}"] = _exp(
            ["pc", "fci"], 30,
            cfg=dict(knowledge_regime="noisy" if r > 0 else "correct", knowledge_noise=r, knowledge_mode="hard"),
            opts=dict(run_ablations=True, knowledge_scope=scope))

# Rfqt-W3 / Kmr8-W2: dose-response in the number of hidden hubs (p = 26 in all).
for h in (0, 1, 2, 3, 4, 6):
    EXPERIMENTS[f"hubdose_h{h}"] = _exp(["pc", "fci"], 20, cfg=dict(HUBS_CFG),
                                        data=dict(n_latent=h, latent_mode="hubs"),
                                        opts=dict(run_ablations=True))

# The same at expected degree 2 (the density of the main setting), so that the
# no-hub arm is a well-specified control of the same size (p = 26).
for h in (0, 2, 4, 6):
    EXPERIMENTS[f"hubdose2_h{h}"] = _exp(["pc", "fci"], 20, cfg=dict(p=26, expected_degree=2.0),
                                         data=dict(n_latent=h, latent_mode="hubs"), opts=dict(run_ablations=True))


def _offset(method: str) -> int:
    return sum(ord(c) * (i + 1) for i, c in enumerate(method)) % 100_000


def run_task(exp_name: str, repeat: int, method: str) -> str:
    spec = EXPERIMENTS[exp_name]
    out = os.path.join(ROOT, exp_name, f"rec_{repeat:03d}_{method}.pkl")
    if os.path.exists(out):
        return f"skip {out}"
    logging.basicConfig(level=logging.WARNING)
    cfg = replace(ExperimentConfig(**BASE_CFG), **spec["cfg"])
    opts = OptionsV3(**spec["opts"])
    seed = cfg.random_seed + repeat
    rng = np.random.default_rng(seed)

    d = spec["data"]
    n_cal = cfg.n - (opts.train_size or int(round(opts.train_frac * cfg.n)))
    size = opts.cal_chunk_size or max(1, n_cal // opts.n_cal_chunks)
    n_test = max(opts.n_test_chunks * size, 2000)
    sc = make_scenario(d.get("scenario", "linear"), cfg.p, cfg.n, n_test, rng,
                       expected_degree=cfg.expected_degree, noise=d.get("noise", "gaussian"),
                       n_latent=d.get("n_latent", 0), latent_mode=d.get("latent_mode", "random"),
                       quadratic_frac=d.get("quadratic_frac", 0.0))
    data, test_data, full_dag, true_graph, weights = sc.data, sc.test_data, sc.full_dag, sc.true_graph, sc.weights

    names = list(data.columns)
    knowledge = make_domain_knowledge(names)
    K = None
    if cfg.knowledge_regime != "none":
        K = knowledge_from_true_graph(true_graph, cfg.knowledge_frac, rng)
        if cfg.knowledge_regime == "noisy":
            K = corrupt_knowledge(K, cfg.knowledge_noise, rng)
        knowledge = K

    t = time.time()
    rec = fit_repeat_v3(data, method, cfg, knowledge, opts, rng_seed=seed * 1000 + _offset(method),
                        true_full_dag=full_dag, true_graph=true_graph, test_data=test_data,
                        soft_knowledge=None, lambda_penalty=0.0, weights=weights)
    rec.update(experiment=exp_name, repeat=repeat, wall=time.time() - t,
               knowledge_regime=cfg.knowledge_regime, knowledge_mode=cfg.knowledge_mode,
               knowledge_scope=opts.knowledge_scope, train_frac=opts.train_frac,
               n_latent=d.get("n_latent", 0),
               knowledge_noise=cfg.knowledge_noise if cfg.knowledge_regime == "noisy" else 0.0,
               knowledge_forbidden=sorted(K.forbidden_edges) if K is not None else None)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(rec, f)
    os.replace(tmp, out)
    return f"done {out} ({rec['wall']:.0f}s)"


def _safe(args):
    try:
        return run_task(*args)
    except Exception:
        return f"FAIL {args}: {traceback.format_exc(limit=4)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k, v in EXPERIMENTS.items():
            print(f"{k:28s} methods={v['methods']} repeats={v['repeats']}")
        return
    tasks = []
    for name in args.experiments:
        spec = EXPERIMENTS[name]
        R = args.repeats or spec["repeats"]
        tasks += [(name, r, m) for r in range(args.start, R) for m in spec["methods"]]
    # Interleave experiments so that partial results cover every arm.
    tasks.sort(key=lambda t: (t[1], t[0], t[2]))
    print(f"{len(tasks)} tasks", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_safe, t) for t in tasks]
        for f in as_completed(futs):
            print(f.result(), flush=True)


if __name__ == "__main__":
    sys.exit(main())
