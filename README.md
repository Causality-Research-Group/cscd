# Conformal Self-Compatible Causal Discovery (CSCD)

Code for the paper *Conformal Self-Compatibility for Domain-Guided Causal Discovery*.

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the paper. causal-learn must be
at least 0.1.4: GES fails in 0.1.3.x.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

The tests cover marginalization (including the chain, fork, collider and
inducing-path cases), invariance across members of an equivalence class, the
distances, subset coverage, hard domain constraints, and the baselines. One
test checks the key property of the method: when every local graph is the
true marginal, the true graph scores exactly zero.

## Reproducing the paper

```bash
./reproduce.sh                # all experiments, then all tables and figures
./reproduce.sh main latent    # selected experiments only
.venv/bin/python make_paper_artifacts.py --results_root results --paper_root ../..
```

`reproduce.sh` lists every experiment and its settings. Results go to
`results/<experiment>/summary.csv`, one row per repeat, base method and alpha.
`make_paper_artifacts.py` writes the LaTeX tables to `../../tables/`, the
figures to `../../figures/results/`, and every mean and standard error to
`results/digest.csv`.

## Code map

| Module | Contents |
| --- | --- |
| `cscd/mixed_graph.py` | Endpoint-mark graphs, member DAGs of CPDAGs and PAGs, marginal MAGs and PAGs |
| `cscd/selfcompat.py` | Subsets with pair coverage, projections, distances, scores, conformal threshold, candidate family |
| `cscd/graph.py` | PC, GES, FCI and DirectLiNGAM wrappers returning endpoint marks |
| `cscd/baselines.py` | Single run, stability selection, uncalibrated self-compatibility, bootstrap compatibility, conformal without subsets |
| `cscd/bayes.py` | Order MCMC baseline |
| `cscd/runner.py` | One repeat of the full pipeline and the experiment drivers |
| `cscd/metrics.py`, `cscd/decision.py` | Claim partition, precision and recall, decision task |

## Graph representation

Graphs are square `pandas.DataFrame`s of endpoint marks. `marks.loc[a, b]` is
the mark at `b` on the edge between `a` and `b`: 0 no edge, 1 tail, 2
arrowhead, 3 circle. A graph over all variables is compared with a graph
learned on a subset `S` only through its **marginal** on `S`
(`selfcompat.project`), never through its induced subgraph. For `A -> B -> C`,
the marginal on `{A, C}` is `A -> C`, whereas the induced subgraph is empty.

Two distances are available with `--distance_mode`:

- `skeleton`: disagreement between the skeletons of the marginal MAGs.
- `pag`: disagreement between the marginal PAGs, counting adjacencies and endpoint marks.

Both depend only on the equivalence class, not on the member used to compute them.
