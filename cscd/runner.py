import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from .config import ExperimentConfig, DomainKnowledge
from .data import (
    generate_random_dag,
    hide_latents,
    load_bnlearn_pgmpy_dataset,
    load_causallearn_sachs,
    load_real_csv,
    sachs_reference_graph,
    simulate_linear_sem,
)
from .utils import (
    corrupt_knowledge,
    knowledge_from_true_graph,
    make_domain_knowledge,
    numpy_to_adj,
)
from .metrics import accepted_precision_recall, shd, skeleton_f1, status_counts, summarize_claims
from .decision import decision_task_summary
from .plotting import make_summary_table, plot_calibration, plot_claim_rates, plot_precision_recall, plot_raw_self_compatibility, plot_score_distributions, plot_top_claims, save_and_plot
from .selfcompat import (
    calibration_scores,
    candidate_graph_score,
    conformal_graph_set,
    conformal_threshold,
    generate_candidate_graphs,
    sample_variable_subsets,
    self_compatibility_score_among_local_graphs,
    train_local_graphs,
)
from .graph import discover_graph
from .baselines import baseline_summary


def _stable_method_offset(method: str) -> int:
    return sum((ord(ch) * (i + 1) for i, ch in enumerate(method))) % 100_000


def _make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _run_one_task(task: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return run_one(**task)


def run_one(
    data: pd.DataFrame,
    knowledge: DomainKnowledge,
    config: ExperimentConfig,
    method: str,
    dataset: str,
    repeat: int,
    true_graph: Optional[pd.DataFrame] = None,
    distance_mode: str = "directed",
    rng_seed: Optional[int] = None,
    soft_knowledge: Optional[DomainKnowledge] = None,
    sem_weights: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if rng_seed is None:
        rng_seed = config.random_seed + repeat * 1000 + _stable_method_offset(method)
    rng = _make_rng(rng_seed)

    lam = config.lambda_penalty if soft_knowledge is not None else 0.0

    var_names = list(data.columns)
    D_tr, D_cal = train_test_split(
        data,
        test_size=0.5,
        shuffle=True,
        random_state=config.random_seed + repeat,
    )
    D_tr = D_tr.reset_index(drop=True)
    D_cal = D_cal.reset_index(drop=True)

    subsets = sample_variable_subsets(
        var_names,
        m=config.n_subsets,
        subset_frac=config.subset_frac,
        rng=rng,
    )

    local_graphs = train_local_graphs(
        D_tr,
        subsets,
        knowledge,
        method,
        config,
        rng,
    )

    raw_self_compat = self_compatibility_score_among_local_graphs(
        local_graphs,
        subsets,
        distance_mode=distance_mode,
    )

    # Compute scores on N + N_test disjoint, equal-sized chunks of D_cal.
    # The first N set the conformal threshold; the held-out remainder provide
    # an honest estimate of the exceedance rate P(A_test > tau_alpha).
    # Evaluating exceedance on the calibration scores themselves would make it
    # a deterministic function of (N, alpha), identical across base methods.
    N_cal = config.n_calibration_resamples
    N_test = config.n_test_chunks
    all_chunk_scores = calibration_scores(
        D_cal,
        subsets,
        local_graphs,
        knowledge,
        method,
        config,
        n_resamples=N_cal + N_test,
        rng=rng,
        distance_mode=distance_mode,
        soft_knowledge=soft_knowledge,
        lambda_penalty=lam,
    )
    cal_scores = all_chunk_scores[:N_cal]
    test_scores = all_chunk_scores[N_cal:]

    candidates = generate_candidate_graphs(
        D_tr,
        knowledge,
        method,
        config,
        config.n_candidate_graphs,
        rng,
    )

    candidate_base_scores = [
        candidate_graph_score(
            G, local_graphs, subsets, distance_mode,
            knowledge=soft_knowledge, lambda_penalty=lam,
        )
        for G in candidates
    ]

    full_graph = discover_graph(
        D_tr,
        method=method,
        knowledge=knowledge,
        config=config,
        random_state=int(rng.integers(1_000_000)),
    )

    baseline_cols = baseline_summary(
        full_graph=full_graph,
        data_train=D_tr,
        knowledge=knowledge,
        method=method,
        config=config,
        rng=rng,
        candidates=candidates,
        candidate_scores=candidate_base_scores,
        raw_self_compat=raw_self_compat,
        var_names=var_names,
        true_graph=true_graph,
    )

    summary_rows = []
    claim_rows = []
    score_rows = []

    for s in cal_scores:
        score_rows.append({
            "dataset": dataset,
            "method": method,
            "repeat": repeat,
            "score_type": "calibration",
            "score": float(s),
        })

    for s in test_scores:
        score_rows.append({
            "dataset": dataset,
            "method": method,
            "repeat": repeat,
            "score_type": "test",
            "score": float(s),
        })

    for s in candidate_base_scores:
        score_rows.append({
            "dataset": dataset,
            "method": method,
            "repeat": repeat,
            "score_type": "candidate",
            "score": float(s),
        })

    for alpha in config.alpha_values:
        tau = conformal_threshold(cal_scores, alpha)
        selected, candidate_scores = conformal_graph_set(
            candidates,
            local_graphs,
            subsets,
            tau,
            distance_mode=distance_mode,
            knowledge=soft_knowledge,
            lambda_penalty=lam,
        )

        claims = summarize_claims(
            selected,
            var_names,
            accepted_threshold=config.accepted_threshold,
            rejected_threshold=config.rejected_threshold,
        )

        counts = status_counts(claims)
        pr = accepted_precision_recall(claims, true_graph)

        # Downstream decision task: estimate treatment-effect bias under
        # four adjustment strategies.  Only available when sem_weights and
        # true_graph are both provided (fully observed synthetic setting).
        _decision_nan: Dict[str, float] = {
            "decision_bias_oracle": float("nan"),
            "decision_bias_single": float("nan"),
            "decision_bias_accepted": float("nan"),
            "decision_bias_notrejected": float("nan"),
            "decision_bias_cscd_avg": float("nan"),
            "decision_bias_naive": float("nan"),
            "decision_n_pairs": float("nan"),
        }
        if sem_weights is not None and true_graph is not None:
            _dec_rng = _make_rng(rng_seed + 7919 + int(alpha * 1000))
            try:
                decision_cols = decision_task_summary(
                    data_eval=data,
                    W=sem_weights,
                    var_names=var_names,
                    true_graph=true_graph,
                    full_graph=full_graph,
                    claims=claims,
                    rng=_dec_rng,
                    n_pairs=config.decision_n_pairs,
                    accepted_threshold=config.accepted_threshold,
                    rejected_threshold=config.rejected_threshold,
                    selected_graphs=selected,
                )
            except Exception as _dec_exc:
                logger.warning("decision_task_summary failed: %s", _dec_exc)
                decision_cols = _decision_nan
        else:
            decision_cols = _decision_nan

        if true_graph is not None:
            graph_shd = shd(full_graph, true_graph)
            sk_prec, sk_rec, sk_f1 = skeleton_f1(full_graph, true_graph)
        else:
            graph_shd = np.nan
            sk_prec, sk_rec, sk_f1 = np.nan, np.nan, np.nan

        # Calibration check (Proposition 1): fraction of *held-out* chunk
        # scores exceeding τ_α should be ≈ α.  tau=inf when k>N, in which case
        # all scores are ≤ τ and the empirical rate is 0.
        finite_tau = np.isfinite(tau)
        if len(test_scores) > 0:
            cal_exceed_rate = float(np.mean(test_scores > tau)) if finite_tau else 0.0
        else:
            cal_exceed_rate = float("nan")
        calibration_error = cal_exceed_rate - alpha
        insample_exceed_rate = float(np.mean(cal_scores > tau)) if finite_tau else 0.0

        total_claims = len(var_names) * (len(var_names) - 1)
        summary_rows.append({
            "dataset": dataset,
            "method": method,
            "repeat": repeat,
            "alpha": alpha,
            "tau": tau if finite_tau else float("nan"),
            "raw_self_compatibility": raw_self_compat,
            "n_selected_graphs": len(selected),
            "candidate_accept_rate": float(np.mean(np.array(candidate_scores) <= tau)),
            "cal_exceed_rate": cal_exceed_rate,
            "insample_exceed_rate": insample_exceed_rate,
            "calibration_error": calibration_error,
            "full_graph_shd": graph_shd,
            "skeleton_precision": sk_prec,
            "skeleton_recall": sk_rec,
            "skeleton_f1": sk_f1,
            **counts,
            **pr,
            "accepted_rate": counts["n_accepted"] / total_claims,
            "rejected_rate": counts["n_rejected"] / total_claims,
            "unresolved_rate": counts["n_unresolved"] / total_claims,
            "knowledge_regime": config.knowledge_regime,
            "knowledge_mode": config.knowledge_mode,
            "lambda_penalty": config.lambda_penalty,
            **baseline_cols,
            **decision_cols,
        })

        claims = claims.copy()
        claims["dataset"] = dataset
        claims["method"] = method
        claims["repeat"] = repeat
        claims["alpha"] = alpha
        claim_rows.append(claims)

    return (
        pd.DataFrame(summary_rows),
        pd.concat(claim_rows, ignore_index=True),
        pd.DataFrame(score_rows),
    )


def _execute_tasks(tasks: List[Dict], n_workers: Optional[int]) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    if n_workers is None or n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_task = {
                executor.submit(_run_one_task, task): task
                for task in tasks
            }
            results = []
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    logger.warning(
                        "Skipping task dataset=%s method=%s repeat=%s: %s",
                        task["dataset"],
                        task["method"],
                        task["repeat"],
                        exc,
                    )
            return results
    results = []
    for task in tasks:
        try:
            results.append(run_one(**task))
        except Exception as exc:
            logger.warning(
                "Skipping task dataset=%s method=%s repeat=%s: %s",
                task["dataset"],
                task["method"],
                task["repeat"],
                exc,
            )
    return results


def run_synthetic(
    methods: List[str],
    config: ExperimentConfig,
    outdir: str,
    distance_mode: str,
    n_workers: Optional[int] = None,
) -> None:
    tasks = []
    all_summary = []
    all_claims = []
    all_scores = []

    for repeat in range(config.n_repeats):
        rng = _make_rng(config.random_seed + repeat)
        B = generate_random_dag(
            config.p,
            config.expected_degree,
            rng,
        )

        data, W_full = simulate_linear_sem(
            B,
            n=config.n,
            noise=config.noise,
            rng=rng,
            return_weights=True,
        )
        data, B_obs = hide_latents(data, B, config.latent_fraction, rng)

        # sem_weights are only passed in the fully observed setting; when
        # latents are hidden the W indexing no longer matches the observed
        # variable order so we pass None to disable the decision task.
        sem_weights_for_task = W_full if config.latent_fraction == 0.0 else None

        var_names = list(data.columns)
        true_graph = numpy_to_adj(B_obs, var_names)

        # Build regime knowledge.
        empty_knowledge = make_domain_knowledge(var_names)
        if config.knowledge_regime == "none":
            regime_knowledge = empty_knowledge
        elif config.knowledge_regime == "correct":
            regime_knowledge = knowledge_from_true_graph(
                true_graph, config.knowledge_frac, rng
            )
        else:  # noisy
            correct_knowledge = knowledge_from_true_graph(
                true_graph, config.knowledge_frac, rng
            )
            regime_knowledge = corrupt_knowledge(
                correct_knowledge, config.knowledge_noise, rng
            )

        # Dispatch per knowledge_mode.
        if config.knowledge_mode == "hard":
            task_knowledge = regime_knowledge
            task_soft_knowledge = None
        else:  # soft
            task_knowledge = empty_knowledge
            task_soft_knowledge = regime_knowledge

        for method in methods:
            tasks.append({
                "data": data,
                "knowledge": task_knowledge,
                "config": config,
                "method": method,
                "dataset": "synthetic",
                "repeat": repeat,
                "true_graph": true_graph,
                "distance_mode": distance_mode,
                "rng_seed": config.random_seed + repeat * 1000 + _stable_method_offset(method),
                "soft_knowledge": task_soft_knowledge,
                "sem_weights": sem_weights_for_task,
            })

    for summary, claims, scores in _execute_tasks(tasks, n_workers):
        all_summary.append(summary)
        all_claims.append(claims)
        all_scores.append(scores)

    save_and_plot(all_summary, all_claims, all_scores, outdir)


def run_named_dataset(
    dataset: str,
    methods: List[str],
    config: ExperimentConfig,
    outdir: str,
    distance_mode: str,
    n_workers: Optional[int] = None,
    real_csv: Optional[str] = None,
) -> None:
    dataset = dataset.lower()

    if dataset == "sachs_real":
        data, ref_graph = load_causallearn_sachs()
        dataset_label = "sachs_real"
    elif dataset == "asia":
        data, ref_graph = load_bnlearn_pgmpy_dataset(
            "asia",
            n_samples=config.n,
            random_seed=config.random_seed,
        )
        dataset_label = "asia"
    elif dataset == "sachs_bn":
        data, ref_graph = load_bnlearn_pgmpy_dataset(
            "sachs",
            n_samples=config.n,
            random_seed=config.random_seed,
        )
        dataset_label = "sachs_bn"
    elif dataset == "child":
        data, ref_graph = load_bnlearn_pgmpy_dataset(
            "child",
            n_samples=config.n,
            random_seed=config.random_seed,
        )
        dataset_label = "child"
    elif dataset == "csv":
        if real_csv is None:
            raise ValueError("--real_csv is required when --dataset csv.")
        data = load_real_csv(real_csv)
        ref_graph = sachs_reference_graph(list(data.columns))
        dataset_label = "csv"
    else:
        raise ValueError(
            "Unknown dataset. Use one of: synthetic, sachs_real, asia, "
            "sachs_bn, child, csv."
        )

    var_names = list(data.columns)
    knowledge = make_domain_knowledge(var_names)

    tasks = []
    for repeat in range(config.n_repeats):
        for method in methods:
            tasks.append({
                "data": data,
                "knowledge": knowledge,
                "config": config,
                "method": method,
                "dataset": dataset_label,
                "repeat": repeat,
                "true_graph": ref_graph,
                "distance_mode": distance_mode,
                "rng_seed": config.random_seed + repeat * 1000 + _stable_method_offset(method),
            })

    all_summary = []
    all_claims = []
    all_scores = []

    for summary, claims, scores in _execute_tasks(tasks, n_workers):
        all_summary.append(summary)
        all_claims.append(claims)
        all_scores.append(scores)

    save_and_plot(all_summary, all_claims, all_scores, outdir)
