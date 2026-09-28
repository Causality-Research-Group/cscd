"""Revision experiments (E1-E14 of the revision plan).

Usage:
    python experiments_v2.py --list
    python experiments_v2.py main_v2 design_N20 ... [--workers 9] [--repeats R]

Each experiment is a named preset.  One record per (experiment, repeat,
method) is pickled to results_v2/<experiment>/rec_<repeat>_<method>.pkl;
existing records are skipped, so runs can be resumed.  Records are analysed
by analyze_v2.py.
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
from typing import Dict, List

import numpy as np

from cscd.config import ExperimentConfig
from cscd.pipeline import PipelineOptions, fit_repeat
from cscd.scenarios import make_scenario
from cscd.utils import corrupt_knowledge, knowledge_from_true_graph, make_domain_knowledge

ROOT = "results_v2"

BASE_CFG = dict(n=3200, p=20, n_subsets=25, subset_frac=0.5, n_candidate_graphs=100,
                stability_n_bootstraps=50, random_seed=2027)


def _exp(methods, repeats, cfg=None, opts=None, data=None, **extra) -> Dict:
    return dict(methods=methods, repeats=repeats, cfg=cfg or {}, opts=opts or {}, data=data or {}, **extra)


LIGHT = dict(run_bayes=False, run_full_data_baselines=False, run_ablations=False)
PCGESFCI = ["pc", "ges", "fci"]

EXPERIMENTS: Dict[str, Dict] = {
    # E1/E3/E5/E11/E12/E13: main operating point with every baseline.
    "main_v2": _exp(PCGESFCI, 30, opts=dict(n_bma_samples=100)),
    # E2: chunk design at n = 3200.
    "design_N20_c40": _exp(PCGESFCI, 20, opts=dict(n_cal_chunks=20, cal_chunk_size=40, **LIGHT)),
    "design_N19": _exp(PCGESFCI, 20, opts=dict(n_cal_chunks=19, **LIGHT)),
    "design_N9": _exp(PCGESFCI, 20, opts=dict(n_cal_chunks=9, **LIGHT)),
    "design_sym_N9": _exp(PCGESFCI, 20, opts=dict(n_cal_chunks=9, train_size=320, **LIGHT)),
    # E2/E8: sample size at N = 9.
    **{f"n{n}_N9": _exp(PCGESFCI if n <= 6400 else ["pc", "fci"], 20 if n >= 1600 else 30,
                        cfg=dict(n=n), opts=dict(n_cal_chunks=9, **LIGHT))
       for n in (200, 400, 800, 1600, 6400, 12800)},
    # E4: systematic errors (with ablations = bootstrap compatibility, no subsets).
    "sys_control": _exp(PCGESFCI, 30, opts=dict(run_bayes=False, run_full_data_baselines=False)),
    "sys_latent_hubs": _exp(PCGESFCI, 30, cfg=dict(p=26, expected_degree=3.0),
                            opts=dict(run_bayes=False, run_full_data_baselines=False),
                            data=dict(n_latent=6, latent_mode="hubs")),
    "sys_highdim": _exp(["pc", "ges"], 20, cfg=dict(p=50, n=1000, n_subsets=50, subset_frac=0.3),
                        opts=dict(run_bayes=False, run_full_data_baselines=False)),
    "sys_wrongK": _exp(["pc", "fci"], 30, cfg=dict(knowledge_regime="noisy", knowledge_noise=0.5),
                       opts=dict(run_bayes=False, run_full_data_baselines=False)),
    "sys_quadratic": _exp(["pc", "fci"], 30, opts=dict(run_bayes=False, run_full_data_baselines=False),
                          data=dict(scenario="quadratic", quadratic_frac=0.5)),
    # E6: subset design.
    **{f"sub_k{k}_m{m}": _exp(["pc", "fci"], 20, cfg=dict(subset_frac=k, n_subsets=m), opts=LIGHT)
       for k in (0.3, 0.5, 0.7) for m in (10, 25, 50)},
    "sub_k0.5_m171": _exp(["pc", "fci"], 20, cfg=dict(subset_frac=0.5, n_subsets=171), opts=LIGHT),
    "sub_k0.7_m33": _exp(["pc", "fci"], 20, cfg=dict(subset_frac=0.7, n_subsets=33), opts=LIGHT),
    "sub_mb_k0.5": _exp(["pc", "fci"], 20, cfg=dict(subset_frac=0.5), opts=dict(subset_mode="mb", **LIGHT)),
    "sub_mb_k0.3": _exp(["pc", "fci"], 20, cfg=dict(subset_frac=0.3), opts=dict(subset_mode="mb", **LIGHT)),
    "sub_k0.3_m10_nocover": _exp(["pc", "fci"], 20, cfg=dict(subset_frac=0.3, n_subsets=10, require_pair_coverage=False), opts=LIGHT),
    # E7: distances (all computed from the same local graphs).
    "distances": _exp(["pc", "fci"], 20, opts=dict(distances=("skeleton", "pag", "skel_extra2", "skel_miss2"), **LIGHT)),
    # E9: PC vs FCI under hidden confounding, as n grows.
    **{f"latent_n{n}": _exp(["pc", "fci"], 20, cfg=dict(p=26, n=n, expected_degree=3.0), opts=LIGHT,
                            data=dict(n_latent=6, latent_mode="hubs"))
       for n in (1600, 6400, 25600)},
    # E10: robustness.
    **{f"knowledge_{r}_{md}": _exp(["pc", "fci"], 20, cfg=dict(knowledge_regime=r, knowledge_mode=md),
                                   opts=dict(run_bayes=False, run_full_data_baselines=False, run_ablations=False))
       for r in ("correct", "noisy") for md in ("hard", "soft")},
    "knowledge_none": _exp(["pc", "fci"], 20, opts=dict(run_bayes=False, run_full_data_baselines=False, run_ablations=False)),
    "large_p50": _exp(["pc", "fci"], 20, cfg=dict(p=50, n_subsets=50), opts=dict(run_bayes=False, run_full_data_baselines=False)),
    "nongauss_laplace": _exp(["pc", "lingam"], 20, opts=dict(run_full_data_baselines=False), data=dict(noise="laplace")),
    "latent_random": _exp(["pc", "fci"], 20, cfg=dict(p=25), opts=dict(run_bayes=False, run_full_data_baselines=False),
                          data=dict(n_latent=5, latent_mode="random")),
    "sachs_obs": _exp(PCGESFCI, 20, cfg=dict(n_subsets=15), opts=dict(n_cal_chunks=9, run_full_data_baselines=False),
                      real="sachs_obs"),
    "sachs_pooled": _exp(PCGESFCI, 20, cfg=dict(n_subsets=15), opts=dict(n_cal_chunks=9, run_full_data_baselines=False),
                         real="sachs_pooled"),
    **{f"bnlearn_{net}": _exp(["pc", "fci"], 20, cfg=dict(n=4000, n_subsets=40, independence_test="gsq"),
                              opts=dict(run_bayes=False, run_full_data_baselines=False), real=net)
       for net in ("asia", "child", "insurance", "alarm")},
    # E12: scaling.
    "scale_p50": _exp(["pc"], 5, cfg=dict(p=50, n_subsets=50), opts=LIGHT),
    "scale_p100": _exp(["pc"], 5, cfg=dict(p=100, n_subsets=100, subset_frac=0.2), opts=LIGHT),
    # E14: nonlinear base test (KCI).
    "kci_quadratic": _exp(["pc"], 10, cfg=dict(p=8, n=900, n_subsets=10, independence_test="kci",
                                            n_candidate_graphs=30, stability_n_bootstraps=20,
                                            candidate_alpha_grid=(0.01, 0.1)),
                          opts=dict(n_cal_chunks=9, n_test_chunks=5, **LIGHT),
                          data=dict(scenario="quadratic", quadratic_frac=1.0)),
}


def _offset(method: str) -> int:
    return sum(ord(c) * (i + 1) for i, c in enumerate(method)) % 100_000


def load_real(name: str, repeat: int, cfg: ExperimentConfig):
    from cscd import data as D

    if name == "sachs_obs":
        return D.load_sachs_observational()
    if name == "sachs_pooled":
        return D.load_causallearn_sachs()
    return D.load_bnlearn_pgmpy_dataset(name, n_samples=cfg.n, random_seed=cfg.random_seed + repeat)


def run_task(exp_name: str, repeat: int, method: str) -> str:
    spec = EXPERIMENTS[exp_name]
    out = os.path.join(ROOT, exp_name, f"rec_{repeat:03d}_{method}.pkl")
    if os.path.exists(out):
        return f"skip {out}"
    logging.basicConfig(level=logging.WARNING)
    cfg = replace(ExperimentConfig(**BASE_CFG), **spec["cfg"])
    opts = PipelineOptions(**spec["opts"])
    seed = cfg.random_seed + repeat
    rng = np.random.default_rng(seed)
    weights, test_data, full_dag, true_graph = None, None, None, None

    if "real" in spec:
        data, true_graph = load_real(spec["real"], repeat, cfg)
        full_dag = true_graph
        names = list(data.columns)
    else:
        d = spec["data"]
        n_test = opts.n_test_chunks * max(1, (cfg.n - (opts.train_size or int(round(opts.train_frac * cfg.n))))
                                          // opts.n_cal_chunks if opts.cal_chunk_size is None else opts.cal_chunk_size)
        n_test = max(n_test, 2000)
        sc = make_scenario(d.get("scenario", "linear"), cfg.p, cfg.n, n_test, rng,
                           expected_degree=cfg.expected_degree, noise=d.get("noise", "gaussian"),
                           n_latent=d.get("n_latent", 0), latent_mode=d.get("latent_mode", "random"),
                           quadratic_frac=d.get("quadratic_frac", 0.0))
        data, test_data, full_dag, true_graph, weights = sc.data, sc.test_data, sc.full_dag, sc.true_graph, sc.weights
        names = list(data.columns)

    knowledge = make_domain_knowledge(names)
    soft, lam = None, 0.0
    if cfg.knowledge_regime != "none" and true_graph is not None:
        K = knowledge_from_true_graph(true_graph, cfg.knowledge_frac, rng)
        if cfg.knowledge_regime == "noisy":
            K = corrupt_knowledge(K, cfg.knowledge_noise, rng)
        if cfg.knowledge_mode == "hard":
            knowledge = K
        else:
            soft, lam = K, cfg.lambda_penalty

    t = time.time()
    rec = fit_repeat(data, method, cfg, knowledge, opts, rng_seed=seed * 1000 + _offset(method),
                     true_full_dag=full_dag, true_graph=true_graph, test_data=test_data,
                     soft_knowledge=soft, lambda_penalty=lam, weights=weights)
    rec.update(experiment=exp_name, repeat=repeat, wall=time.time() - t,
               knowledge_regime=cfg.knowledge_regime, knowledge_mode=cfg.knowledge_mode)
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
        return f"FAIL {args}: {traceback.format_exc(limit=3)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--repeats", type=int, default=None, help="Override the number of repeats.")
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
        tasks += [(name, r, m) for r in range(R) for m in spec["methods"]]
    print(f"{len(tasks)} tasks", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_safe, t) for t in tasks]
        for f in as_completed(futs):
            print(f.result(), flush=True)


if __name__ == "__main__":
    sys.exit(main())
