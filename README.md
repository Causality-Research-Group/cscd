# Conformal Self-Compatible Causal Discovery (CSCD)

Code for the paper *Conformal Self-Compatibility for Domain-Guided Causal Discovery*.

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins the versions used for the paper. causal-learn must be
at least 0.1.4: GES fails in 0.1.3.x. The permutation-test detector uses DoWhy
(`pip install dowhy==0.14`); DoWhy declares `scipy<=1.15.3`, but runs with the
pinned scipy 1.17.1, which we keep so that all other results are unchanged.

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
./reproduce.sh            # every experiment, then every table and figure
./reproduce.sh tables     # tables and figures only, from existing records
```

Two resumable drivers hold the presets (`--list` prints them):
`experiments_v2.py` (design sweeps, robustness, bnlearn) writes records to
`results_v2/`, and `experiments_v3.py` (validity study, main experiment with
candidate-family arms, multi-chain order MCMC and external detectors,
detection scenarios, domain knowledge, alpha sweep, scalability,
semi-synthetic ecoli70, Sachs with interventional validation) writes to
`results_v3/`. Records hold raw objects (all local graphs, scores, candidates,
baseline outputs), so every number is recomputed from them by `analyze_v2.py`
and `analyze_v3.py`; `make_paper_tables_v2.py`, `make_tables_v3.py`,
`make_figures_v2.py` and `make_figures_v3.py` write the paper's tables and
figures. `diagnostics/knowledge_trace.py` reproduces the diagnosis of the hard
domain-knowledge constraint.

## Code map

| Module | Contents |
| --- | --- |
| `cscd/mixed_graph.py` | Endpoint-mark graphs, member DAGs of CPDAGs and PAGs, marginal MAGs and PAGs |
| `cscd/selfcompat.py` | Subsets with pair coverage, projections, distances, scores, conformal threshold, candidate family |
| `cscd/graph.py` | PC, GES, FCI and DirectLiNGAM wrappers returning endpoint marks |
| `cscd/baselines.py` | Single run, stability selection, uncalibrated self-compatibility, bootstrap compatibility, conformal without subsets |
| `cscd/bayes.py` | Order MCMC baseline |
| `cscd/pipeline.py`, `cscd/pipeline_v3.py` | One repeat of the full pipeline, storing a record |
| `cscd/detectors.py` | Faller et al.'s kappa_G, the permutation test of Eulig et al. (DoWhy), LOVO of Schkoda et al. |
| `cscd/analysis_v3.py` | Claim types, matched-recall precision, inclusion diagnostics of Theorem 1, detection metrics |
| `cscd/sachs.py` | Sachs interventional conditions as held-out validation |
| `cscd/runner.py` | Drivers of the first experiments |
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
