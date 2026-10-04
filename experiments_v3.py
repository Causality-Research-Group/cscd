"""Second-revision experiments (review items R1-R11; see REVISION_V3.md).

Usage:
    python experiments_v3.py --list
    python experiments_v3.py valid_N9 main_v3 ... [--workers 9] [--repeats R]

One record per (experiment, repeat, method) is pickled to
results_v3/<experiment>/rec_<repeat>_<method>.pkl; existing records are
skipped, so runs can be resumed.  Records are analysed by analyze_v3.py.
Seeds follow experiments_v2.py, so repeat r of a synthetic preset draws the
same graph and data as repeat r of the corresponding v2 preset.
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
from cscd.scenarios import make_scenario, make_semisynthetic
from cscd.utils import corrupt_knowledge, knowledge_from_true_graph, make_domain_knowledge

ROOT = "results_v3"

BASE_CFG = dict(n=3200, p=20, n_subsets=25, subset_frac=0.5, n_candidate_graphs=100,
                stability_n_bootstraps=50, random_seed=2027)
PCGESFCI = ["pc", "ges", "fci"]
LIGHT = dict(run_ablations=False)
DETECT = dict(run_ablations=True, run_external=True)


def _exp(methods, repeats, cfg=None, opts=None, data=None, **extra) -> Dict:
    return dict(methods=methods, repeats=repeats, cfg=cfg or {}, opts=opts or {}, data=data or {}, **extra)


EXPERIMENTS: Dict[str, Dict] = {
    # R5: large-repeat validity check (scores only), with many held-out chunks
    # so that the conditional exceedance of each repeat can be estimated.
    "valid_N9": _exp(PCGESFCI, 500, opts=dict(score_only=True, n_cal_chunks=9, n_test_chunks=40)),
    "valid_N19": _exp(PCGESFCI, 300, opts=dict(score_only=True, n_cal_chunks=19, n_test_chunks=40)),
    "valid_N4": _exp(PCGESFCI, 300, opts=dict(score_only=True, n_cal_chunks=4, n_test_chunks=40)),
    # R1/R2/R3/R6/R7/R11: main operating point, every baseline, the candidate-pool
    # arms, multi-chain order MCMC, external detectors (also the detection control).
    "main_v3": _exp(PCGESFCI, 30, opts=dict(pool="arms", run_bayes=True, mcmc_iterations=20000, run_full_data_baselines=True,
                                            run_decision=True, **DETECT)),
    # R5: alpha sweep of the set (alpha = 0.2 -> N = 4, alpha = 0.05 -> N = 19).
    "alpha_N4": _exp(PCGESFCI, 30, opts=dict(n_cal_chunks=4, **LIGHT)),
    "alpha_N19": _exp(PCGESFCI, 30, opts=dict(n_cal_chunks=19, **LIGHT)),
    # R3: systematic-error scenarios with all detectors.
    "hubs_v3": _exp(PCGESFCI, 30, cfg=dict(p=26, expected_degree=3.0), opts=DETECT,
                    data=dict(n_latent=6, latent_mode="hubs")),
    "highdim_v3": _exp(PCGESFCI, 20, cfg=dict(p=50, n=1000, n_subsets=50, subset_frac=0.3, n_candidate_graphs=40),
                       opts=DETECT),
    "lowdim_n1000_v3": _exp(PCGESFCI, 20, cfg=dict(p=20, n=1000, n_subsets=50, subset_frac=0.75, n_candidate_graphs=40),
                            opts=DETECT),
    "quadratic_v3": _exp(["pc", "fci"], 30, opts=DETECT, data=dict(scenario="quadratic", quadratic_frac=0.5)),
    # R9: domain knowledge with the corrected hard constraint, a sweep of the
    # fraction of reversed constraints, and the soft penalty (calibration with
    # and without the penalty is stored in the same record).
    **{f"know_hard_rev{int(100 * r)}": _exp(["pc", "fci"], 20, cfg=dict(knowledge_regime="noisy" if r > 0 else "correct",
                                                                        knowledge_noise=r, knowledge_mode="hard"),
                                            opts=dict(run_ablations=True))
       for r in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)},
    "know_none": _exp(["pc", "fci"], 20, opts=dict(run_ablations=True)),
    **{f"know_soft_{reg}": _exp(["pc", "fci"], 20, cfg=dict(knowledge_regime=reg, knowledge_mode="soft"), opts=LIGHT)
       for reg in ("correct", "noisy")},
    # R10: scalability with Markov-blanket subsets.
    "scale_p100_mb": _exp(["pc"], 5, cfg=dict(p=100, subset_frac=0.15, n_candidate_graphs=50, stability_n_bootstraps=20),
                          opts=dict(subset_mode="mb", n_boot_baseline=20, **LIGHT)),
    "scale_p200_mb": _exp(["pc"], 5, cfg=dict(p=200, n=5000, subset_frac=0.1, n_candidate_graphs=50,
                                              stability_n_bootstraps=20),
                          opts=dict(subset_mode="mb", n_boot_baseline=20, **LIGHT)),
    # R8: semi-synthetic benchmark and Sachs with interventional validation.
    "semi_ecoli70": _exp(["pc", "ges"], 20, cfg=dict(n_subsets=40), opts=dict(run_bayes=True, mcmc_iterations=20000, **LIGHT),
                         data=dict(scenario="semi", network="ecoli70")),
    "sachs_obs_v3": _exp(PCGESFCI, 20, cfg=dict(n_subsets=15), opts=dict(n_test_chunks=3, run_bayes=True, mcmc_iterations=20000, **LIGHT),
                         real="sachs_obs"),
}


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
    weights, test_data, full_dag, true_graph = None, None, None, None

    if "real" in spec:
        from cscd import data as D

        data, true_graph = D.load_sachs_observational()
        full_dag = true_graph
    else:
        d = spec["data"]
        n_cal = cfg.n - (opts.train_size or int(round(opts.train_frac * cfg.n)))
        size = opts.cal_chunk_size or max(1, n_cal // opts.n_cal_chunks)
        n_test = max(opts.n_test_chunks * size, 2000)
        if d.get("scenario") == "semi":
            sc = make_semisynthetic(d["network"], cfg.n, n_test, rng)
        else:
            sc = make_scenario(d.get("scenario", "linear"), cfg.p, cfg.n, n_test, rng,
                               expected_degree=cfg.expected_degree, noise=d.get("noise", "gaussian"),
                               n_latent=d.get("n_latent", 0), latent_mode=d.get("latent_mode", "random"),
                               quadratic_frac=d.get("quadratic_frac", 0.0))
        data, test_data, full_dag, true_graph, weights = sc.data, sc.test_data, sc.full_dag, sc.true_graph, sc.weights

    names = list(data.columns)
    knowledge = make_domain_knowledge(names)
    soft, lam, K = None, 0.0, None
    if cfg.knowledge_regime != "none" and true_graph is not None:
        K = knowledge_from_true_graph(true_graph, cfg.knowledge_frac, rng)
        if cfg.knowledge_regime == "noisy":
            K = corrupt_knowledge(K, cfg.knowledge_noise, rng)
        if cfg.knowledge_mode == "hard":
            knowledge = K
        else:
            soft, lam = K, cfg.lambda_penalty

    t = time.time()
    rec = fit_repeat_v3(data, method, cfg, knowledge, opts, rng_seed=seed * 1000 + _offset(method),
                        true_full_dag=full_dag, true_graph=true_graph, test_data=test_data,
                        soft_knowledge=soft, lambda_penalty=lam, weights=weights)
    rec.update(experiment=exp_name, repeat=repeat, wall=time.time() - t,
               knowledge_regime=cfg.knowledge_regime, knowledge_mode=cfg.knowledge_mode,
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
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k, v in EXPERIMENTS.items():
            print(f"{k:24s} methods={v['methods']} repeats={v['repeats']}")
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
