
## 1. Status of every plan item

✅ done · 🟡 done with caveats · ⏳ still running at the time of writing (re-run `make_report.sh` to refresh) · ❌ not done

### Experiments

| Item | What | Status | Where |
|---|---|---|---|
| E1 | No-filter ablation and selectivity | ✅ | §2.1, App. A.1–A.2 |
| E2 | Chunk design (N = 20/19/9, symmetric), sample-size path | ✅ | §2.2, App. A.4 |
| E3 | Matched-recall comparison with baselines (D_train and full D, MB error bound, bootstrap frequency, order MCMC once per repeat) | ✅ | §2.1, App. A.1, A.3 |
| E4 | Systematic-error detection vs bootstrap compatibility / no subsets | ✅ | §2.3, App. A.5 |
| E5 | Bootstrap compatibility at median and (1−α) quantile, same knowledge | ✅ (merged into E1/E4) | App. A.1, A.5 |
| E6 | Subset design grid, Prop. 5 m, Markov-blanket subsets, no pair cover | ✅ | §2.4, App. A.6 |
| E7 | Distances: skeleton, PAG, asymmetric weightings | 🟡 the gadjid-style ancestral distance was not implemented (see §4) | §2.4, App. A.7 |
| E8 | Low sample (n = 200–1600) and safeguards | ✅ | §2.4, App. A.4 |
| E9 | PC vs FCI under hidden confounding, ε_j against n (1600–25 600) | 🟡 ran; the predicted PC/FCI gap is **not** observed (see §2.6) | §2.6, App. A.8 |
| E10 | Knowledge, p = 50, Laplace/LiNGAM, latent, Sachs (observational and pooled), bnlearn | 🟡 all done; Child/Insurance/Alarm with a reduced protocol (§4) | §2.6, App. A.9 |
| E11 | Decision task, redone (local IDA, held-out data, set/bootstrap/BMA averaging) | ✅ | §2.5, App. A.10 |
| E12 | Runtime and scaling (p = 50, 100) | ✅ | §2.6 |
| E13 | γ and α sensitivity | ✅ | App. A.2 |
| E14 | Nonlinear base test (PC + KCI, p = 8) | ✅ (10 repeats) | §2.6, App. A.9 |

### Method, theory and writing (paper repository, local branch `experiments/chris`)

| Item | Status | Notes |
|---|---|---|
| M1 chunk design | ✅ | Default is now N = ⌈1/α⌉ − 1 = 9 chunks with a 50/50 split. The symmetric design was tested and rejected (it loses G\*). |
| M2 orientation-aware claims | 🟡 | PAG distance evaluated (E7): no gain. Adjacency-level claims are now reported next to directed claims. |
| M3 Markov-blanket subsets | ✅ | `cscd/pipeline.py::markov_blanket_subsets` |
| T1 finite-sample rate | ✅ | Corollary `cor:kb-rate`. Kalisch & Bühlmann rate for PC on causally sufficient subsets; qualitative for FCI. |
| T2 exclusion / chunk design | ✅ | Corollary `cor:exclusion`, Remark `rem:chunkdesign` |
| T3 detectable systematic errors | ✅ | Proposition `prop:detect`. The example was corrected after E4 (see §4). |
| T4 MAG marginal vs ADMG latent projection | ✅ | §2.1 remark, based on the verified reading of Faller et al. (2024) |
| W1 abstract/conclusion without the AI-evaluation hook | ✅ | New abstract with real numbers (§5) |
| W2 intro contributions | ✅ | Three contributions, including a candid statement on claim precision |
| W3 related-work comparison table | ✅ | `tab:related`, with 9 verified new references |
| W4 marginalisation remark | ✅ | |
| W5 practical recipe | ✅ | "Choosing the design" paragraph in §2 |
| W6 distance properties | ✅ | |
| W7 low-sample safeguards | ✅ | Results §4.4. The admitted fraction is the diagnostic to report. |
| W8 consistency fixes | ✅ | FCI-vs-PC overclaim removed; G\* scored separately (stated); Bayesian baseline once; Sachs observational; failure counts recorded |
| W9 scope | ✅ | Robustness grid and decision task moved to the appendix; main text is exactly 8 pages |

## 2. Findings in more detail

### 2.1 Calibration, selectivity and claims (E1, E3, E5, E13)

- **Calibration.** Held-out exceedance at α = 0.1 is 0.133 ± 0.029 (PC), 0.043 ± 0.014 (GES) and 0.113 ± 0.021 (FCI) over 30 repeats; at α = 0.2 it is 0.21 / 0.15 / 0.18.
  - With N = 9, a threshold at α = 0.05 is +∞ by construction (it needs N ≥ 19); `design_N19` covers that level.
  - Held-out and calibration scores have the same distribution: the mean rank of held-out scores among calibration scores is 0.50 (PC) and 0.49 (FCI).
- **Selectivity.**
  - Admitted fraction: 0.59 (PC), 0.42 (GES), 0.78 (FCI).
  - G\* retained: 100% for all three.
  - The full-data run itself is excluded in 7% (PC), 17% (GES) and 3% (FCI) of repeats.
- **The no-filter ablation is decisive.** At γ = 0.9, CSCD's accepted claims are no more precise than a vote over the unfiltered candidate pool, at any γ from 0.6 to 1.0 (App. A.2).
  - For PC and FCI they are statistically indistinguishable.
  - For GES the unfiltered pool is clearly better (directed AP 0.91 vs 0.69), because the filter removes the sparse `max_parents = 1` grid candidates that make the vote conservative.
- **Adjacency level** (the level at which Prop. 4 makes its promise).
  - CSCD: 0.983 / 0.881 (PC), 0.904 / 0.970 (GES), 0.995 / 0.901 (FCI) precision / recall.
  - Stability selection at 0.6: 0.980 / 0.909, 0.935 / 0.966, 0.965 / 0.934.
  - Order MCMC: 0.90–0.92 precision.
- **Directed level.** Stability selection, bootstrap frequency and order MCMC are more precise at equal recall.
- **Full-data baselines** (the fairness issue A3 in the plan). Giving the baselines the full sample changes their precision by at most about 0.05, and does not change the conclusions (App. A.1).

### 2.2 Chunk design (E2)

| n = 3200 | N × n_c | τ (PC) | Admitted PC / GES / FCI | G\* retained |
|---|---|---|---|---|
| old | 20 × 40 | 0.109 | 0.87 / 0.53 / 0.995 | 1.00 |
| | 19 × 84 | 0.084 | 0.75 / 0.44 / 0.96 | 1.00 |
| **new default** | 9 × 177 | 0.064 | 0.59 / 0.37 / 0.77 | 1.00 |
| symmetric | 9 × 320, n_tr = 320 | 0.048 | 0.26 / 0.10 / 0.48 | 0.55 / 1.00 / 0.60 |

This confirms Remark `rem:chunkdesign` and Theorem 1 quantitatively. Along the sample-size path at N = 9, τ keeps falling as n grows (PC: 0.135 at n = 200 to 0.059 at n = 6400), and admission falls with it (0.95 to 0.47).

### 2.3 Systematic errors (E4): the new headline

| Scenario | Method | Skeleton errors of Ĝ | CSCD rejects Ĝ | Bootstrap compat. rejects Ĝ | AUROC CSCD / Boot. | G\* retained (CSCD / Boot.) |
|---|---|---|---|---|---|---|
| Control | PC / GES / FCI | 2.8 / 3.0 / 2.3 | 0.07 / 0.17 / 0.03 | 0 / 0 / 0 | – | 1.00 / 0.90; 1.00 / 0.97; 1.00 / 0.97 |
| Hidden hubs | PC | 36.0 | **0.90** | 0.00 | **0.95** / 0.61 | 0.33 / 0.07 |
| | GES | 33.7 | **0.87** | 0.00 | **0.91** / 0.13 | 0.97 / 0.33 |
| | FCI | 36.6 | 0.33 | 0.00 | 0.64 / 0.55 | 0.43 / 0.03 |
| p = 50, n = 1000 | GES | 22.3 | **0.89** | 0.00 | **0.94** / 0.37 | 1.00 / 1.00 |
| | PC | 7.5 | 0.15 | 0.00 | 0.65 / **0.94** | 0.95 / 1.00 |
| Wrong hard knowledge | PC / FCI | 4.5 / 2.9 | 0.20 / 0.10 | 0 / 0 | 0.69 / 0.49; 0.64 / 0.44 | 0.77 / 0.67; 1.00 / 0.93 |
| Quadratic (shared bias) | PC / FCI | 14.1 / 14.5 | 0.00 / 0.00 | 0 / 0 | 0.22 / 0.07; 0.02 / 0.06 | 0.30 / 0.47; 0.20 / 0.50 |

How to read this:
- **Case (a) of Proposition `prop:detect` works.** The full-variable error depends on which variables are analysed together (hidden hubs for PC and GES, greedy GES at p = 50). The subset check flags it in about 90% of runs; bootstrap compatibility never does. The no-subset conformal ablation behaves like bootstrap compatibility: it admits Ĝ in 100% of runs.
- **CSCD also keeps G\* much more often** than bootstrap compatibility under hidden hubs.
- **Case (b) is invisible, as predicted.** With the quadratic bias shared by every subset, AUROC is below 0.5 because the biased graphs are sparser. There bootstrap compatibility keeps G\* more often (0.47–0.50 vs 0.20–0.30).
- **Two caveats.** First, PC at p = 50 is a case where bootstrap compatibility is the better detector (AUROC 0.94 vs 0.65). Second, FCI's hidden-hub errors are flagged only 33% of the time. Both are stated in the paper.
- **A measurement fix.** "Skeleton errors" are now measured against the true MAG skeleton, which includes bidirected edges. Before the fix, the table compared against the directed claims only and over-counted FCI errors.

### 2.4 Subsets, distances, low sample (E6, E7, E8)

- **Separation cover grows much faster with k than with m.**
  - k = 0.3p: 0.87 (m = 10) → 0.92 (m = 50).
  - k = 0.5p: 0.95 (m = 10) → 0.985 (m = 25).
  - k = 0.7p: 0.997 (m = 10).
  - Prop. 5's m ≈ 171 for k = 10 is very conservative: m = 25 plus pair coverage already gives 0.985.
- **Markov-blanket subsets** reach 0.945 cover at k = 0.3p with about 27 subsets, against 0.89 for 25 random ones.
- **Claim quality is flat across designs.** Runtime is roughly proportional to the number of subsets (17 s at m = 10 up to 147 s at m = 171).
- **Distances.**
  - PAG mode keeps calibration (exceedance 0.105 FCI) but lowers G\* inclusion to 0.90 without a precision gain.
  - Penalising extra edges more loses G\* (0.70–0.80).
  - Penalising missing edges more keeps G\* and slightly improves PC's AP (0.751 vs 0.726).
  - Skeleton stays the default.
- **Low sample.**
  - Exceedance stays valid down to n = 200 (chunks of 11 samples).
  - No empty sets appear, except GES in 3–15% of repeats at n = 400–1600.
  - The filter is uninformative at n ≤ 400 (admits 90–99% for PC and FCI), because ε_j(n_c) ≈ 1.
  - Accepted directed claims stay more precise than the single run at every n: PC 0.68 vs 0.62 at n = 200.
  - Recommendation (paper §4.4): report the admitted fraction. When it is close to 1, the calibrated check carries no information beyond the candidate pool.

### 2.5 Decision task (E11)

Setup: local IDA on held-out data, over the confounded pairs of the true DAG. Repeats with no confounded pair (8 of 30) are excluded.

| | Oracle | No adj. | Single | CSCD set | Bootstrap | BMA |
|---|---|---|---|---|---|---|
| PC, mean abs. bias | 0.035 | 0.461 | 0.428 | 0.401 | 0.371 | 0.372 |
| PC, coverage of true effect | – | – | 0.23 | 0.63 | 0.70 | 0.38 |

Paired differences (set − single): PC −0.019 ± 0.008, GES −0.006 ± 0.011, FCI −0.022 ± 0.014. Bootstrap averaging is comparable (better for PC, worse for GES). The honest summary: a small bias reduction, a large coverage gain, and no advantage over bootstrap averaging.

### 2.6 Hidden variables, robustness, real data, runtime (E9, E10, E12, E14)

- **E9: hidden hubs, PC vs FCI.** Theorem 1 predicts that FCI retains G\* more often than PC once local errors vanish. That regime is **not reached** up to n = 25 600:
  - ε̂_j (the fraction of subsets with at least one wrong adjacency) is 0.97 → 0.89 (PC) and 0.97 → 0.91 (FCI);
  - A(G\*) falls from 0.20 to 0.14 for both;
  - G\* is retained in 30–43% of repeats for both.

  The paper now says this plainly. The earlier sentence claiming the effect was measured is gone.
- **Random hidden variables (5 of 25).** Exceedance is 0.08 (PC) and 0.13 (FCI); G\* is retained 95% of the time.
- **Domain knowledge.**
  - Calibration holds in every regime; the largest exceedance is 0.18 ± 0.04 (PC, correct soft knowledge).
  - Correct hard knowledge raises directed precision (PC 0.74 → 0.81). Noisy hard knowledge lowers G\* inclusion to 0.85.
  - Soft knowledge makes the filter permissive (FCI admits 97%). The penalty also enters the calibration scores, and noisy chunk graphs violate it, which inflates τ to 0.15–0.23.
- **p = 50, Laplace noise.** Calibration holds (exceedance 0.06–0.13). With DirectLiNGAM on Laplace noise, accepted directed claims are 0.994 precise, against 0.981 for the single run.
- **Sachs, observational (853 cells).**
  - Exceedance is 0.07–0.15.
  - Accepted adjacencies are **all** in the consensus network (adjacency precision 1.00, recall 0.17–0.32), but directed precision is below the single run (PC 0.36 vs 0.44).
  - The 17-edge consensus network is **never** admitted: A(G\*) ≈ 0.37 against τ ≈ 0.14, which is worth a sentence in the paper.
  - The pooled data (7466 cells) are reported only as a stress test.
- **bnlearn.**
  - Asia (full protocol): exceedance 0.07 (PC) and 0.12 (FCI); G\* retained in 95–100% of repeats.
  - Child, Insurance and Alarm use the reduced protocol (App. A.9).
- **E14: KCI on quadratic data** (p = 8, n = 900, 10 repeats). Exceedance 0.16 ± 0.09 with only 5 held-out chunks; G\* retained in 90% of repeats; directed precision 0.67 vs 0.61 for the single run. A nonlinear base test removes the shared bias of case (b).
- **E12: runtime** (PC, one core).
  - About 78 s per repeat at p = 20 with every baseline; about 28 s for CSCD alone (m = 25).
  - CSCD alone takes 364 s at p = 50 (m = 50) and about 1520 s at p = 100 (108 subsets of size 20).
  - Cost is dominated by m (N + 1 + held-out) local runs plus the candidate projections.
