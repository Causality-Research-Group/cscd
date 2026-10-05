"""Add the true MAG skeleton ('true_skel') to records written before it was stored.

The data of every synthetic repeat is regenerated from its seed exactly as in
experiments_v2.run_task; real-data records use the reference graph.
"""
import glob
import os
import pickle
import sys
from dataclasses import replace

import numpy as np

import experiments_v2 as E
from cscd.config import ExperimentConfig
from cscd.mixed_graph import NO_EDGE, from_dag_adjacency
from cscd.pipeline import PipelineOptions
from cscd.scenarios import make_scenario
from cscd.selfcompat import project

for path in sorted(glob.glob(os.path.join(E.ROOT, "*", "rec_*.pkl"))):
    with open(path, "rb") as f:
        rec = pickle.load(f)
    if rec.get("true_skel") is not None:
        continue
    spec = E.EXPERIMENTS[rec["experiment"]]
    names = rec["var_names"]
    if "real" in spec:
        A = rec["true_graph"]
        rec["true_skel"] = ((A + A.T) > 0).astype(np.int8)
    else:
        cfg = replace(ExperimentConfig(**E.BASE_CFG), **spec["cfg"])
        opts = PipelineOptions(**spec["opts"])
        rng = np.random.default_rng(cfg.random_seed + rec["repeat"])
        d = spec["data"]
        n_test = opts.n_test_chunks * max(1, (cfg.n - (opts.train_size or int(round(opts.train_frac * cfg.n))))
                                          // opts.n_cal_chunks if opts.cal_chunk_size is None else opts.cal_chunk_size)
        n_test = max(n_test, 2000)
        sc = make_scenario(d.get("scenario", "linear"), cfg.p, cfg.n, n_test, rng,
                           expected_degree=cfg.expected_degree, noise=d.get("noise", "gaussian"),
                           n_latent=d.get("n_latent", 0), latent_mode=d.get("latent_mode", "random"),
                           quadratic_frac=d.get("quadratic_frac", 0.0))
        assert list(sc.data.columns) == names
        assert (sc.true_graph.loc[names, names].values == rec["true_graph"]).all(), path
        sk = project(from_dag_adjacency(sc.full_dag), names, "skeleton")
        rec["true_skel"] = (sk.loc[names, names].values != NO_EDGE).astype(np.int8)
    tmp = path + ".bf"
    with open(tmp, "wb") as f:
        pickle.dump(rec, f)
    os.replace(tmp, path)
    print("backfilled", path, flush=True)
