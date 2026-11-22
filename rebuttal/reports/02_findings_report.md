# Rebuttal experiments, wave 2: findings (22–23 Nov 2026)

Branch `cscd/rebuttal/experiments-chris` (from `rebuttal_exp` at 9b4539a; nothing on the professor's branches was changed).
Every number below comes from `rebuttal/out/chris/*.csv` (per-run files next to each summary), which `rebuttal/analyze_chris.py` regenerates from the records in `results_v3_chris/` (local only, gitignored) and `results_v3/`.
α = 0.1, N = 9, γ = 0.9 unless stated. Intervals are Wilson 95% over independent runs.

> **Read §1 first.** One result affects a sentence already in the new abstract and in the responses to Rfqt, xFQE and Kmr8.

---

## 1. Matched control for hidden hubs (Rfqt-W3, xFQE-W3, Kmr8-W2; new abstract). **Changes a claim in the draft**

The draft abstract and responses say: *"the threshold flags 87–90% of PC and GES runs [with hidden confounders] … and 7–17% of runs without confounders."*
The 7–17% comes from `main_v3` (p = 20, expected degree 2). The hidden-hub runs use a **denser** graph (p = 26, expected degree 3, 6 hubs hidden → 20 observed). The comparison therefore mixes density with confounding.

**New control `sinks_v3`.** Same seeds, so the same 30 DAGs and data as `hubs_v3`, but the 6 hidden variables are sinks. This gives the same density and the same 20 observed variables, with **no** hidden confounding (`latent_mode="sinks"`, `cscd/scenarios.py`).

| Method | Setting | Skel. err. of Ĝ (pairs) | CSCD flag | Bootstrap flag (own quantile) | G* kept |
|---|---|---|---|---|---|
| PC | hidden hubs (paper) | 36.0 | 0.90 [0.74, 0.97] | 0.00 | 0.33 |
| PC | hidden sinks (paired control) | 4.1 | **0.47 [0.30, 0.64]** | 0.00 | 0.70 |
| PC | main setting (paper's control) | 2.8 | 0.07 [0.02, 0.21] | 0.00 | 1.00 |
| GES | hidden hubs | 33.7 | 0.87 [0.70, 0.95] | 0.00 | 0.97 |
| GES | hidden sinks | 5.9 | **0.30 [0.17, 0.48]** | 0.00 | 1.00 |
| GES | main setting | 3.0 | 0.17 [0.07, 0.34] | 0.00 | 1.00 |
| FCI | hidden hubs | 36.6 | 0.33 [0.19, 0.51] | 0.00 | 0.43 |
| FCI | hidden sinks | 4.4 | 0.13 [0.05, 0.30] | 0.00 | 0.67 |

AUROC of the CSCD margin, hubs vs paired sinks: **PC 0.87, GES 0.86, FCI 0.54** (bootstrap compatibility: 0.37, 0.19, 0.37).

What this means:
- Detection holds against the fair control for PC and GES (AUROC 0.86–0.87), but at its own threshold the false-flag rate is **30–47%**, not 7–17%.
- These "false" flags are not empty. Among the sink runs, flagged runs have 2–3 times the skeleton error of kept runs (PC 6.2 vs 2.3; GES 11.8 vs 3.3; FCI 9.0 vs 3.7). In dense graphs, PC/GES subset graphs err systematically even without hidden variables, and the flag picks up those runs. Pooled over hubs and sinks, kept runs have skeleton error 4.4 (PC) / 7.7 (GES), against 27.3 / 28.4 for flagged runs.
- The dose-response series (§6) gives the same picture. At degree 3, CSCD flags 85% of PC runs even with **0** hidden hubs (Ĝ already has 9.5 errors). At degree 2, hiding 6 hubs only triples Ĝ's error (3.4 → 12.0), and the flag rate rises only from 0.10 to 0.20 (PC).

**Suggested rewording** (abstract/responses): "In simulations with hidden confounders, the threshold flags 87–90% of PC and GES runs; at the same graph density without confounders, it flags 30–47%, and the runs it flags there have two to three times the skeleton error of those it keeps. Bootstrap resampling with its own threshold flags none." Or frame the flag as a detector of *systematic full-run error* (whatever its cause) rather than of confounding.

---

## 2. AI-3.1: the redefined "Conformal, no subsets" baseline (promised rerun, done)

Two non-degenerate versions on the **same data, split and Ĝ** as the stored records (regenerated Ĝ equals the stored one in 98% of the 360 runs; the rest are FCI tie-breaks):
- *full-data*: R = Ĝ = A(D_train); calibration d(A(chunk_b), R); tested graph A(D). This is the reading of the draft response, but it is not strictly conformal, because A(D) uses the calibration rows.
- *split*: R = A(D_train,a); tested graph A(D_train,b); calibration on the same 9 chunks. This version is valid.

Native flag with each detector's own threshold (360 runs; `nosub.csv`):

| Scenario | Method | CSCD | No subsets, full-data | No subsets, split | AUROC no-subsets (full / split) |
|---|---|---|---|---|---|
| Hidden hubs | PC | 0.90 | 0.00 | 0.00 | 0.12 / 0.14 |
| Hidden hubs | GES | 0.87 | 0.00 | 0.03 | 0.08 / 0.18 |
| Hidden hubs | FCI | 0.33 | 0.00 | 0.00 | 0.11 / 0.21 |
| p = 50, n = 1000 | GES | 0.90 | 0.00 | 0.00 | 0.88 / 0.85 |
| Quadratic | PC / FCI | 0.00 / 0.00 | 0.00 / 0.00 | 0.07 / 0.10 | 0.22 / 0.30, 0.21 / 0.44 |
| Controls (main, p=20 n=1000) | all | 0–0.17 | 0.00 | 0–0.07 | — |

The no-subsets baseline almost never fires. Under hidden hubs its AUROC is **below 0.25**: confounded runs look *more* stable as full-variable graphs. This is the cleanest answer to "what do subsets add" (Rfqt-Q2, xFQE-W2): the detection comes from the subsets, and calibration alone adds none. It replaces the degenerate column of Tables 2/10 (the one with 0.00 everywhere). Caveat to state: on p = 50, raw no-subset distances separate scenario from control (AUROC 0.85–0.99), but that is dimensionality, as the paper already says for every raw score.

---

## 3. Kmr8-Q1: training fraction 0.3–0.8 (promised sweep, done)

Score-only runs, main setting, 30 runs per cell (`trainfrac.csv`):

| Method | n_train/n | chunk | τ₀.₁ | exceedance | G* kept | Δ̂ | ε̂ train | flag on Ĝ |
|---|---|---|---|---|---|---|---|---|
| PC | 0.3 | 248 | 0.056 | 0.087 | **0.87 [0.70, 0.95]** | 0.006 | 0.72 | 0.20 |
| PC | 0.4 | 213 | 0.060 | 0.108 | 1.00 | 0.013 | 0.72 | 0.13 |
| PC | **0.5** | 177 | 0.065 | 0.115 | 1.00 | 0.019 | 0.68 | 0.07 |
| PC | 0.6 | 142 | 0.072 | 0.060 | 1.00 | 0.026 | 0.66 | 0.07 |
| PC | 0.8 | 71 | 0.090 | 0.085 | 1.00 | 0.047 | 0.62 | 0.00 |
| GES | 0.3 → 0.8 | | 0.086 → 0.137 | 0.08–0.10 | 1.00 everywhere | 0.024 → 0.076 | | 0.33 → 0.07 |
| FCI | 0.3 | 248 | 0.054 | 0.067 | **0.90 [0.74, 0.97]** | 0.005 | 0.65 | 0.07 |
| FCI | 0.4 | 213 | 0.057 | 0.088 | 0.93 | 0.012 | 0.65 | 0.03 |
| FCI | 0.5 → 0.8 | | 0.061 → 0.088 | 0.09–0.12 | 1.00 | 0.018 → 0.046 | | 0.03 → 0.00 |

Answer to Kmr8:
- Validity holds at every split (pooled exceedance 0.08–0.11).
- Theorem 1(iii) at work: a larger training share lowers training error and shrinks the chunks, so the margin Δ̂ grows eightfold from 0.3 to 0.8 and inclusion rises.
- The price is selectivity. τ grows by 60% for PC and FCI and by 59% for GES from 0.3 to 0.8, and the flag on Ĝ fires less often (part of this is that Ĝ itself is fitted on more data).
- **0.5 is the smallest training share at which G\* is kept in every run for all three methods**, so it keeps the tightest threshold compatible with full inclusion. That is the justification for 50/50.

---

## 4. d2sj-W4: sample-size scaling, PC vs FCI, with and without hidden confounders (promised FCI study, done)

Score-only, 30 runs per cell, n = 400 → 25,600, fixed α_test = 0.05 (`nscale.csv`). The αₙ → 0 version is in §4b.

| Scenario | Method | n | subsets with any error ε̂ | wrong (subset, pair) slots | extra / missing per slot | G* kept | flag on Ĝ | Skel. err. Ĝ |
|---|---|---|---|---|---|---|---|---|
| no latents | FCI | 400 | 0.78 | 0.055 | 0.010 / 0.045 | 1.00 | 0.00 | 3.9 |
| no latents | FCI | 25,600 | 0.60 | 0.024 | 0.014 / 0.011 | 1.00 | 0.07 | 1.5 |
| no latents | PC | 25,600 | 0.56 | 0.023 | 0.012 / 0.011 | 1.00 | 0.30 | 1.4 |
| 6 hidden hubs | FCI | 400 | 0.98 | 0.247 | 0.006 / 0.240 | 0.27 | 0.00 | 41.9 |
| 6 hidden hubs | FCI | 25,600 | 0.92 | 0.132 | 0.006 / 0.126 | 0.37 | 0.77 | 32.2 |
| 6 hidden hubs | PC | 25,600 | 0.91 | 0.128 | 0.007 / 0.122 | 0.37 | 1.00 | 30.4 |

P(τ = 0) = P(A(G*) = 0) = 0 in every cell.

This is an **honest negative** for the response as drafted ("the event of Theorem 2 becomes likely"):
- Up to n = 25,600 the event of Theorem 2 never occurs. Two reasons: (a) with a fixed test level, about 1.2% of pair slots stay spurious adjacencies at every n, which is the type-I floor (consistency needs αₙ → 0; see §4b); (b) under dense hidden hubs, subset graphs **miss** weak adjacencies of the true marginal MAG (inducing paths through hidden hubs), and this falls only slowly (24% → 13% of slots).
- FCI behaves like PC on both counts, so being latent-aware does not change the picture at these sample sizes.
- What does scale is the **flag**: under hidden hubs it rises from 0% to 77% (FCI) and from 3% to 100% (PC) as n grows, while G* inclusion stays at 0.27–0.37.

**Candidate family under hidden hubs** (`nscand.csv`, 15 runs per cell, standard 101-graph family):

| Method | n | admitted fraction | set empty | false adj. per admitted graph (mean / max) | missing adj. per admitted graph | flag on Ĝ |
|---|---|---|---|---|---|---|
| FCI | 1,600 | 0.80 | 0.00 | 1.1 / 4.2 | 35.5 | 0.07 |
| FCI | 25,600 | 0.11 | 0.67 | 1.0 / 2.8 | 12.2 | 0.73 |
| PC | 1,600 | 0.17 | 0.27 | 0.7 / 2.2 | 35.2 | 0.73 |
| PC | 25,600 | 0.003 | 0.93 | 0.0 / 0.0 | 4.2 | 1.00 |

This matches Proposition 2 / Theorem 2 qualitatively: admitted graphs have about one false adjacency or none, the error is in **missing** edges, and with more data CSCD abstains (empty set) rather than admit wrong graphs.

### 4b. Shrinking test level αₙ = 0.05·1600/n (n = 6,400 → 102,400; 30 runs per cell)

| Scenario | Method | n | ε̂ train | extra / missing per slot | P(A(G*) = 0) | P(τ = 0) | G* kept | flag on Ĝ | Skel. err. Ĝ |
|---|---|---|---|---|---|---|---|---|---|
| no latents | FCI | 6,400 | 0.43 | 0.0026 / 0.017 | 0.00 | 0.00 | 0.93 | 0.13 | 1.6 |
| no latents | FCI | 25,600 | 0.31 | 0.0006 / 0.014 | 0.20 | 0.00 | 0.90 | 0.30 | 0.9 |
| no latents | FCI | 102,400 | **0.25** | 0.0005 / 0.010 | 0.20 | 0.07 | 0.93 [0.79, 0.98] | 0.27 | 0.7 |
| no latents | PC | 102,400 | 0.23 | 0.0004 / 0.008 | 0.27 | 0.03 | 0.83 [0.66, 0.93] | 0.37 | 0.6 |
| hidden hubs | FCI | 102,400 | 0.83 | 0.0003 / 0.127 | 0.00 | 0.00 | 0.20 | 0.97 | 31.5 |
| hidden hubs | PC | 102,400 | 0.85 | 0.0005 / 0.116 | 0.00 | 0.00 | 0.10 | 1.00 | 29.9 |

- Shrinking the test level removes the spurious-edge floor (1.2% → 0.05% of slots). Without latents, local error then falls steadily, and the event of Theorem 2 starts to appear (A(G*) = 0 in 20–27% of runs, τ = 0 in up to 7%). Convergence is slow, though: missing weak edges still dominate at n = 102,400.
- Under hidden hubs, nothing changes with n for FCI or PC. About 12% of pair slots are still missed, inclusion is 0.10–0.20, and the flag fires in 97–100% of runs.
- Suggested answer to d2sj-W4: the threshold is valid at every n and for every learner (pooled exceedance 0.097 over 840 runs; per cell 0.05–0.16, each cell resting on only 30 runs × 10 held-out chunks). The reliability of the passing set follows the local error, and that error vanishes only with αₙ → 0 and without dense hidden confounding. In that regime even FCI does not reach the limit at n ≤ 10⁵, and what the user gets is a flag/abstention, not a correct set. This separates "calibration validity" from "screening reliability", as the reviewer asked.

---

## 5. Rfqt-Q1: what the framework adds to domain knowledge

### 5a. Scope of hard knowledge (`kscope.csv`, 20 runs; the "all" and "none" arms are `results_v3/know_*`)

| Arm (PC) | admitted | flag on Ĝ | AUROC vs no knowledge | orient. precision / recall | single-run orient. prec. / rec. |
|---|---|---|---|---|---|
| no knowledge | 0.59 | 0.05 | — | 0.93 / 0.47 | 0.83 / 0.58 |
| K everywhere (paper), correct | 0.64 | 0.05 | 0.46 | 0.96 / 0.59 | 0.88 / 0.66 |
| K in full-variable runs only, correct | 0.63 | 0.05 | 0.45 | 0.96 / 0.59 | 0.88 / 0.66 |
| K everywhere, 30% reversed | 0.61 | 0.05 | 0.52 | 0.92 / 0.52 | 0.84 / 0.60 |
| K in full-variable runs only, 30% reversed | 0.60 | 0.05 | 0.51 | 0.90 / 0.53 | 0.83 / 0.62 |

FCI: the same pattern; orientation recall stays below 0.06 in every arm.

Answer to Rfqt-Q1, plainly:
- Hard constraints in this code only orient edges. In skeleton mode (the default score) they leave every subset skeleton, and hence the score and the threshold, **unchanged**: training subset graphs with and without K have identical skeletons, and calibration scores agree to the 4th decimal.
- Hence "knowledge in subset graphs" adds nothing in skeleton mode. Knowledge in the candidates gives all the gain (PC orientation precision 0.93 → 0.96, recall 0.47 → 0.59), and that is exactly what giving K to the base algorithm does.
- Wrong knowledge is invisible in every scope (AUROC 0.45–0.65).
- To make the framework's own contribution real, the score would need PAG mode (orientation marks) or the soft penalty (§5b).

### 5b. Soft penalty λ (never tuned before; offline from `results_v3/know_soft_*`, 20 runs; `lam.csv`)

| PC, correct K | λ | τ | G* kept | admitted | flag on Ĝ | adj. recall | orient. prec. |
|---|---|---|---|---|---|---|---|
| penalty in A_b (paper) | 0 → 1 | 0.064 → **0.152** | 1.00 | 0.59 → **0.76** | 0.05 | 0.88 → 0.91 | 0.93 → 0.95 |
| penalty on candidates only | 0 → 1 | 0.064 | 1.00 | 0.59 → 0.17 | 0.05 → 0.70 | 0.88 → 0.55 | 0.93 |
| … 30% reversed K, candidates only | 1 | 0.064 | **0.55** | 0.11 | 0.80 | 0.41 | 0.91 |

Findings:
1. **As implemented in the paper, the soft penalty is nearly inert.** Chunk subset graphs, learned without knowledge, violate constraints by chance, so λR_K inflates τ more than it penalises candidates. The admitted fraction *grows* with λ (PC 0.59 → 0.76; FCI 0.77 → 0.97) and claims barely change.
2. With the penalty on candidates only (τ calibrated without it), the set shrinks with λ. With correct K, G* is always kept, which is guaranteed: R_K(G*) = 0, so A_λ(G*) = A(G*). With 30% reversed K, G* is lost in 45% of PC runs at λ = 1, so this version *can* expose wrong soft knowledge, at the cost of admitting fewer graphs.
3. R_K counts an **undirected** edge as violating both directions (via `to_adjacency`). Candidates are therefore penalised for leaving edges unoriented, which is why adjacency recall falls with λ. This looks like a bug: R_K should count only definite directed edges.

Suggested answer to Rfqt-Q1: say honestly that hard knowledge acts through the base algorithm. The framework's specific contribution is the soft penalty, and with the penalty applied to candidates and the threshold calibrated without it, the guarantee for G* is unchanged when K is correct, while wrong K can cost G*. This would require fixing R_K, and it supports Rfqt's remark that "Domain-Guided" overstates the hard-knowledge part.

---

## 6. Dose-response in the number of hidden hubs (p = 26; `hubdose*.csv`)

| Degree | Hidden hubs | PC skel. err. | PC CSCD flag | FCI CSCD flag | Bootstrap flag |
|---|---|---|---|---|---|
| 3 | 0 | 9.5 | 0.85 | 0.45 | 0.00 |
| 3 | 2 | 23.8 | 0.85 | 0.45 | 0.00 |
| 3 | 6 | 37.8 | 0.90 | 0.25 | 0.00 |
| 2 | 0 | 3.4 | 0.10 | 0.05 | 0.00 |
| 2 | 2 | 6.9 | 0.15 | 0.05 | 0.00 |
| 2 | 6 | 12.0 | 0.20 | 0.00 | 0.00 |

There is no clean dose-response. The flag tracks density (PC/GES subset errors are systematic in dense graphs) more than the number of hidden confounders. Gating still pays off: pooled over hub levels, kept PC runs have skeleton error 10.9 vs 25.7 for flagged runs (degree 3) and 5.4 vs 18.7 (degree 2). The flag is precise but has low recall at degree 2. FCI is rarely flagged at either density.

---

## 7. d2sj-W1 / Kmr8-W1: claim error as a function of γ (offline, `main_v3`; `gamma.csv`)

Orientation (definite) claims, main setting, FDP over runs with a claim (recall in parentheses):

| γ | PC CSCD | PC pool | PC bootstrap (D_train) | GES CSCD | GES bootstrap |
|---|---|---|---|---|---|
| 0.9 | 0.041 (0.51) | 0.057 (0.42) | 0.046 (0.36) | 0.139 (0.65) | 0.056 (0.61) |
| 1.0 | 0.013 (0.09) | 0.000 (0.02) | 0.025 (0.22) | 0.073 (0.20) | 0.003 (0.51) |

For PC, CSCD's orientation claims reach the bootstrap's FDP with 40% more recall at γ = 0.9. For GES, they are worse at every γ, consistent with the FDP table already in the draft.

---

## 8. Small code issues found (not fixed on the professor's branches)

1. `cscd/utils.py::corrupt_knowledge` picks the reversed constraints by iterating a Python `set` (`list(knowledge.forbidden_edges)`). The choice depends on the string hash seed, so wrong-knowledge runs are not reproducible run by run. This is the same class of bug the professor fixed in `decision._descendants`. Fix: `forbidden = sorted(knowledge.forbidden_edges)`.
2. `knowledge_penalty` counts undirected edges as violations (§5b.3).
3. Soft penalty in the calibration scores inflates τ (§5b.1).

## 9. What was not done

- The degree-3/degree-2 hub series used PC and FCI only (GES at p = 26 is ~5× slower); `sinks_v3` has GES.
- Tail of the queue (n = 800/3,200/12,800 for the scaling, h = 1/4, the no-latent candidate-family scaling): *[status at the end of the session — see §10]*.
- The machine was shared with other applications for most of the run, so per-task wall times were 3–10× longer than nominal. The training-fraction sweep was therefore run score-only (no admitted fraction or claims), and repeat counts are 20–30 per cell.

## 10. Reproduce

```bash
.venv/bin/python -m rebuttal.exp_chris --list
WORKERS=8 rebuttal/run_chris.sh          # all experiments, resumable
.venv/bin/python -W ignore -m rebuttal.nosub_chris --workers 8
.venv/bin/python -W ignore -m rebuttal.analyze_chris   # writes rebuttal/out/chris/*.{csv,md}
```
