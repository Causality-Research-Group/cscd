"""Trace why correct hard knowledge lowers G* inclusion (review issue 9).

Regenerates the data, knowledge, split and subsets of the knowledge_correct_hard
records and compares local skeleton errors on D_train for three variants:
no knowledge, the original hard constraint (deletes a forbidden definite edge),
and the fixed constraint (adds an arrowhead, never deletes an adjacency).
"""
import glob, pickle, sys
from dataclasses import replace
import numpy as np, pandas as pd
from cscd.config import ExperimentConfig
from cscd.graph import discover_marks
from cscd.mixed_graph import from_dag_adjacency, NO_EDGE, ARROW, has_edge
from cscd.scenarios import make_scenario
from cscd.selfcompat import project, graph_distance
from cscd.utils import knowledge_from_true_graph, make_domain_knowledge, restrict_knowledge, forbids_edge
import cscd.graph as G
from experiments_v2 import BASE_CFG, EXPERIMENTS

def fixed_constraints(marks, K):
    marks = marks.copy(); v = list(marks.index)
    for a in v:
        for b in v:
            if a != b and has_edge(marks, a, b) and forbids_edge(K, a, b):
                marks.loc[b, a] = ARROW
    return marks

exp = sys.argv[1] if len(sys.argv) > 1 else "knowledge_correct_hard"
spec = EXPERIMENTS[exp]
cfg = replace(ExperimentConfig(**BASE_CFG), **spec["cfg"])
rows = []
for path in sorted(glob.glob(f"results_v2/{exp}/rec_*.pkl")):
    rec = pickle.load(open(path, "rb"))
    r, meth = rec["repeat"], rec["method"]
    rng = np.random.default_rng(cfg.random_seed + r)
    n_test = 2000
    sc = make_scenario("linear", cfg.p, cfg.n, n_test, rng, expected_degree=cfg.expected_degree)
    assert np.array_equal(sc.true_graph.values, rec["true_graph"]), "regeneration mismatch"
    K = knowledge_from_true_graph(sc.true_graph, cfg.knowledge_frac, rng)
    D_tr = sc.data.iloc[rec["_D_tr_index"]].reset_index(drop=True)
    tm = from_dag_adjacency(sc.full_dag)
    empty = make_domain_knowledge(list(sc.data.columns))
    for S in rec["subsets"]:
        truth = project(tm, S, "skeleton")
        raw = discover_marks(D_tr[S], meth, empty, cfg)
        KS = restrict_knowledge(K, S)
        old = G.apply_hard_domain_constraints(raw, KS)
        new = fixed_constraints(raw, KS)
        e = lambda g: graph_distance(project(g, S, "skeleton"), truth, "skeleton")
        n_del = int(((raw.values != NO_EDGE) & (old.values == NO_EDGE)).sum() // 2)
        rows.append(dict(repeat=r, method=meth, err_none=e(raw) > 0, err_old=e(old) > 0, err_new=e(new) > 0,
                         d_none=e(raw), d_old=e(old), d_new=e(new), deleted=n_del))
df = pd.DataFrame(rows)
print(df.groupby("method")[["err_none", "err_old", "err_new", "d_none", "d_old", "d_new", "deleted"]].mean().round(4).to_string())
# subsets where old knowledge creates an error the raw output does not have
print("subsets made wrong by old K:", df.groupby("method").apply(lambda g: int(((~g.err_none) & g.err_old).sum())).to_dict(),
      "of", df.groupby("method").size().to_dict())
df.to_csv(f"diagnostics/knowledge_trace_{exp}.csv", index=False)
