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
from typing import List, Tuple

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
