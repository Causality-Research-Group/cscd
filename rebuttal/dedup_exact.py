"""Exact deduplication of FCI candidates (answer to d2sj-W3).

Usage:  python -m rebuttal.dedup_exact RECORD_DIR [--pag]

The deduplication key of the paper (``mec_key``: skeleton and unshielded
colliders of the marginal MAG over V) is exact for DAG classes but can merge
two MAG classes that differ only in a collider on a discriminating path.
Such a merge can only happen inside a group of admitted candidates that share
the old key but are different graphs.  We therefore test only those groups,
with an exact criterion:

* default: the Markov-equivalence criterion of Spirtes and Richardson (1996):
  two MAGs are Markov equivalent if and only if they have the same
  adjacencies, the same unshielded colliders, and, for every path that is
  discriminating for a vertex V in both graphs, V is a collider on it in one
  graph exactly when it is in the other;
* ``--pag``: the PAG of each graph (``mec_key_pag``), several minutes per
  graph.  Used as a cross-check.  It agreed in 29 of 30 runs; in repeat 23 it
  merged two graphs that the criterion separates, and a direct d-separation
  test showed the criterion right (X6 and X4 are separated by {X3, X14} in
  one graph and not in the other).

When a group splits, the claims are recomputed with the refined classes.
"""

import collections
import glob
import os
import pickle
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from cscd.analysis_v3 import claim_frequencies, claims_row, dedup_weights, standard_mask, truth_claims
from cscd.mixed_graph import ARROW, NO_EDGE, TAIL, marginal_mag
from cscd.pipeline_v3 import mec_key_pag
from cscd.selfcompat import conformal_threshold

ALPHA, GAMMA = 0.1, 0.9


def discriminating_paths(mag: np.ndarray) -> Dict[Tuple[int, ...], bool]:
    """Every discriminating path of a MAG, mapped to whether V is a collider on it.

    ``mag[a, b]`` is the mark at b on the edge a *-* b.  A path
    <X, Q_1, ..., Q_k, V, Y> (k >= 1) is discriminating for V when X and Y are
    not adjacent and every Q_i is a collider on the path and a parent of Y.
    The returned key is the path as a tuple of vertices, from X to Y.
    """
    p = mag.shape[0]
    adj = mag != NO_EDGE

    def is_parent(a, b):  # a -> b
        return mag[a, b] == ARROW and mag[b, a] == TAIL

    out: Dict[Tuple[int, ...], bool] = {}
    for y in range(p):
        for v in np.flatnonzero(adj[y]):
            # Walk backwards from V: the path so far is [current, ..., V, Y].
            stack = [[w, v] for w in np.flatnonzero(adj[v]) if w != y and is_parent(w, y) and mag[v, w] == ARROW]
            while stack:
                path = stack.pop()
                q = path[0]  # a collider candidate and parent of Y
                for z in np.flatnonzero(adj[q]):
                    if z in path or z == y or mag[z, q] != ARROW:
                        continue  # q must be a collider: arrowhead at q from z
                    if not adj[z, y]:
                        full = tuple([z] + path + [y])
                        w = path[-2]
                        out[full] = bool(mag[w, v] == ARROW and mag[y, v] == ARROW)
                    elif is_parent(z, y) and mag[q, z] == ARROW:
                        # z becomes the next collider: it must be a parent of Y
                        # and have an arrowhead from q (the arrowhead from its
                        # other neighbour is checked when the path is extended).
                        stack.append([z] + path)
    return out


def equivalent_mags(m1: np.ndarray, m2: np.ndarray) -> bool:
    """Spirtes-Richardson criterion; assumes equal adjacencies and unshielded colliders."""
    d1, d2 = discriminating_paths(m1), discriminating_paths(m2)
    return all(d1[u] == d2[u] for u in d1.keys() & d2.keys())


def refine(groups: Dict[bytes, List[int]], same) -> Dict[int, int]:
    """Split each old-key group into classes with the pairwise test ``same``."""
    label, next_label = {}, 0
    for members in groups.values():
        reps: List[int] = []
        for i in members:
            for r in reps:
                if same(i, r):
                    label[i] = label[r]
                    break
            else:
                reps.append(i)
                label[i] = next_label
                next_label += 1
    return label


def evaluate(rec, use_pag: bool) -> Dict:
    names = rec["var_names"]
    tau = conformal_threshold(np.asarray(rec["cal_scores"]["skeleton"]), ALPHA)
    scores = np.asarray(rec["cand_scores"]["skeleton"])
    admitted = np.flatnonzero(standard_mask(rec["cand_tags"]) & (scores <= tau))
    row = dict(method=rec["method"], repeat=rec["repeat"], n_admitted=len(admitted))
    if len(admitted) == 0:
        return row

    # Identical graphs are trivially in the same class; test only distinct ones.
    groups: Dict[bytes, List[int]] = collections.defaultdict(list)
    first_of: Dict[bytes, int] = {}
    for i in admitted:
        b = rec["cand_marks"][i].tobytes()
        if b not in first_of:
            first_of[b] = i
            groups[rec["cand_mec"][i]].append(i)

    cache: Dict[int, object] = {}

    def frame(i):
        return pd.DataFrame(rec["cand_marks"][i], index=names, columns=names)

    if use_pag:
        def same(i, j):
            for k in (i, j):
                if k not in cache:
                    cache[k] = mec_key_pag(frame(k), names)
            return cache[i] == cache[j]
    else:
        def same(i, j):
            for k in (i, j):
                if k not in cache:
                    cache[k] = marginal_mag(frame(k), names).loc[names, names].values
            return equivalent_mags(cache[i], cache[j])

    multi = {k: v for k, v in groups.items() if len(v) > 1}
    label = refine(multi, same)
    row["groups_tested"] = len(multi)
    row["graphs_tested"] = sum(len(v) for v in multi.values())
    row["groups_split"] = sum(len({label[i] for i in v}) > 1 for v in multi.values())

    old_keys = [rec["cand_mec"][i] for i in admitted]
    new_keys = []
    for i, key in zip(admitted, old_keys):
        rep = first_of[rec["cand_marks"][i].tobytes()]
        # Candidates outside the tested groups keep their old key.
        new_keys.append(key + b"#%d" % label[rep] if rep in label else key)
    row["classes_old"], row["classes_exact"] = len(set(old_keys)), len(set(new_keys))
    marks = rec["cand_marks"][admitted]
    for kind in ("adj", "dir_def"):
        truth = truth_claims(rec["true_graph"], rec.get("true_skel"), kind)
        for tag, keys in (("old", old_keys), ("exact", new_keys)):
            r = claims_row(claim_frequencies(marks, kind, dedup_weights(keys)), truth, kind, GAMMA)
            row[f"{kind}_prec_{tag}"] = r["prec"]
    return row


def main(record_dir: str, use_pag: bool) -> None:
    rows = []
    for path in sorted(glob.glob(os.path.join(record_dir, "rec_*_fci.pkl"))):
        with open(path, "rb") as f:
            rec = pickle.load(f)
        rows.append(evaluate(rec, use_pag))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    os.makedirs("rebuttal/out", exist_ok=True)
    df.to_csv(f"rebuttal/out/dedup_exact_{'pag' if use_pag else 'sr'}.csv", index=False)
    print(df[["groups_tested", "graphs_tested", "groups_split", "classes_old", "classes_exact"]].sum().to_string())
    for kind in ("adj", "dir_def"):
        print(kind, "mean prec old/exact:", df[f"{kind}_prec_old"].mean().round(4), df[f"{kind}_prec_exact"].mean().round(4))


if __name__ == "__main__":
    main(sys.argv[1], "--pag" in sys.argv)
