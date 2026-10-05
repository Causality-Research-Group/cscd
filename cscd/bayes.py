"""Bayesian structure learning baseline: order MCMC (Friedman and Koller, 2003).

The posterior over DAGs is approximated by sampling topological orders with
Metropolis-Hastings.  Given an order, each node's parent set is summed out
exactly over all sets of predecessors, which gives the order score and lets
DAGs be drawn from the order-conditional posterior.

Local scores are linear-Gaussian BIC scores (a Laplace approximation to the
log marginal likelihood), with a uniform prior over parent sets.  To keep the
enumeration small, each node's parents are restricted to its
``max_candidates`` most correlated variables and to at most ``max_parents``
parents, as is standard for order MCMC (e.g. Kuipers et al., 2022).

The output is a matrix of posterior edge probabilities P(a -> b | data).
"""

import itertools
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp


def _bic_local_score(cov: np.ndarray, n: int, child: int, parents: Tuple[int, ...]) -> float:
    """BIC of a linear-Gaussian regression of ``child`` on ``parents`` (higher is better)."""
    if parents:
        idx = list(parents)
        beta = np.linalg.solve(cov[np.ix_(idx, idx)], cov[idx, child])
        residual_var = cov[child, child] - cov[child, idx] @ beta
    else:
        residual_var = cov[child, child]
    residual_var = max(residual_var, 1e-12)
    return -0.5 * n * np.log(residual_var) - 0.5 * np.log(n) * (len(parents) + 1)


class OrderMCMC:
    """Order MCMC sampler over DAGs for one dataset."""

    def __init__(self, data: pd.DataFrame, max_parents: int = 3, max_candidates: int = 10):
        X = data.values - data.values.mean(axis=0)
        self.names = list(data.columns)
        self.p = X.shape[1]
        n = X.shape[0]
        cov = X.T @ X / n
        corr = np.abs(np.corrcoef(X, rowvar=False))
        np.fill_diagonal(corr, -1.0)

        # For each node: candidate parent sets, their scores, and bit masks.
        self.parent_sets: List[List[Tuple[int, ...]]] = []
        self.scores: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        for child in range(self.p):
            candidates = list(np.argsort(-corr[child])[:max_candidates])
            sets = [
                tuple(sorted(s))
                for size in range(max_parents + 1)
                for s in itertools.combinations(candidates, size)
            ]
            self.parent_sets.append(sets)
            self.scores.append(np.array([_bic_local_score(cov, n, child, s) for s in sets]))
            self.masks.append(np.array([sum(1 << int(v) for v in s) for s in sets], dtype=np.int64))

    def _allowed(self, child: int, predecessors_mask: int) -> np.ndarray:
        """Which parent sets of ``child`` use only predecessors in the order."""
        return (self.masks[child] & ~np.int64(predecessors_mask)) == 0

    def order_log_score(self, order: List[int]) -> float:
        total = 0.0
        predecessors = 0
        for child in order:
            total += logsumexp(self.scores[child][self._allowed(child, predecessors)])
            predecessors |= 1 << child
        return total

    def sample_dag(self, order: List[int], rng: np.random.Generator) -> np.ndarray:
        """Draw a DAG from the posterior given an order."""
        adj = np.zeros((self.p, self.p), dtype=int)
        predecessors = 0
        for child in order:
            allowed = np.flatnonzero(self._allowed(child, predecessors))
            logits = self.scores[child][allowed]
            probs = np.exp(logits - logsumexp(logits))
            chosen = self.parent_sets[child][allowed[rng.choice(len(allowed), p=probs)]]
            for parent in chosen:
                adj[parent, child] = 1
            predecessors |= 1 << child
        return adj

    def edge_posterior(
        self,
        rng: np.random.Generator,
        n_iterations: int = 4000,
        burn_in: int = 1000,
        thin: int = 10,
    ) -> pd.DataFrame:
        """Posterior edge probabilities from a single Metropolis-Hastings chain.

        Proposals swap two random positions of the order (symmetric).
        """
        order = list(rng.permutation(self.p))
        current = self.order_log_score(order)
        counts = np.zeros((self.p, self.p))
        n_samples = 0
        for it in range(n_iterations):
            i, j = rng.choice(self.p, size=2, replace=False)
            proposal = order.copy()
            proposal[i], proposal[j] = proposal[j], proposal[i]
            proposed = self.order_log_score(proposal)
            if np.log(rng.random()) < proposed - current:
                order, current = proposal, proposed
            if it >= burn_in and (it - burn_in) % thin == 0:
                counts += self.sample_dag(order, rng)
                n_samples += 1
        return pd.DataFrame(counts / max(n_samples, 1), index=self.names, columns=self.names)


def bayesian_edge_posterior(
    data: pd.DataFrame,
    rng: np.random.Generator,
    max_parents: int = 3,
    max_candidates: int = 10,
    n_iterations: int = 4000,
) -> pd.DataFrame:
    sampler = OrderMCMC(data, max_parents=max_parents, max_candidates=max_candidates)
    return sampler.edge_posterior(rng, n_iterations=n_iterations, burn_in=n_iterations // 4)


def _split_rhat(chains: np.ndarray) -> float:
    """Split-R-hat (Gelman et al., 2013) of scalar traces, shape (n_chains, n_draws)."""
    n = chains.shape[1] // 2
    if n < 2:
        return float("nan")
    halves = np.concatenate([chains[:, :n], chains[:, n:2 * n]], axis=0)
    W = halves.var(axis=1, ddof=1).mean()
    B = n * halves.mean(axis=1).var(ddof=1)
    if W <= 0:
        return 1.0 if B <= 0 else float("inf")
    var_plus = (n - 1) / n * W + B / n
    return float(np.sqrt(var_plus / W))


def multi_chain_order_mcmc(
    data: pd.DataFrame,
    rng: np.random.Generator,
    n_chains: int = 4,
    n_iterations: int = 10000,
    burn_in: Optional[int] = None,
    thin: int = 25,
    max_parents: int = 3,
    max_candidates: int = 10,
) -> dict:
    """Order MCMC with several independent chains and convergence diagnostics.

    Each chain starts from an independent random order.  Returns the pooled
    edge posterior, the per-chain edge posteriors, the split-R-hat of the
    order log-score traces, the largest absolute difference of an edge
    marginal between any two chains, and the pooled DAG samples
    (``n_chains * (n_iterations - burn_in) / thin`` of them).
    """
    if burn_in is None:
        burn_in = n_iterations // 4
    sampler = OrderMCMC(data, max_parents=max_parents, max_candidates=max_candidates)
    p = sampler.p
    traces, per_chain, samples = [], [], []
    for _ in range(n_chains):
        order = list(rng.permutation(p))
        current = sampler.order_log_score(order)
        trace, dags = [], []
        for it in range(n_iterations):
            i, j = rng.choice(p, size=2, replace=False)
            proposal = order.copy()
            proposal[i], proposal[j] = proposal[j], proposal[i]
            proposed = sampler.order_log_score(proposal)
            if np.log(rng.random()) < proposed - current:
                order, current = proposal, proposed
            if it >= burn_in and (it - burn_in) % thin == 0:
                trace.append(current)
                dags.append(sampler.sample_dag(order, rng))
        traces.append(trace)
        dags = np.stack(dags).astype(np.int8)
        per_chain.append(dags.mean(axis=0))
        samples.append(dags)
    per_chain = np.stack(per_chain)
    diffs = [np.abs(per_chain[a] - per_chain[b]).max() for a in range(n_chains) for b in range(a + 1, n_chains)]
    return {
        "posterior": per_chain.mean(axis=0),
        "per_chain": per_chain,
        "rhat_logscore": _split_rhat(np.array(traces)),
        "max_edge_diff": float(max(diffs)) if diffs else 0.0,
        "samples": np.concatenate(samples, axis=0),
        "names": sampler.names,
    }
