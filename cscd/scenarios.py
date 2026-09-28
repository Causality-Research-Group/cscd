"""Data-generating scenarios for the revision experiments.

Every scenario returns a ``Scenario`` with the observed data, a *fresh*
held-out sample from the same model (used only to estimate the exceedance
rate and for the decision task, never by the method), the full
data-generating DAG, the true parent claims among the observed variables,
and the linear weights when the model is linear and fully observed.

Scenarios
---------
linear      linear SEM, Gaussian/uniform/Laplace noise, optional hidden
            variables chosen at random (``latent_mode='random'``) or among
            the variables with the largest out-degree (``latent_mode='hubs'``,
            which creates many hidden common causes).
quadratic   additive SEM in which a fraction of the edges act through a
            centred quadratic, so that Fisher-z tests miss them on every
            variable subset alike (a bias shared by all local discoveries).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .data import generate_random_dag, topological_order, true_observed_graph


@dataclass
class Scenario:
    data: pd.DataFrame
    test_data: pd.DataFrame
    full_dag: pd.DataFrame
    true_graph: pd.DataFrame
    weights: Optional[np.ndarray]
    name: str


def random_weights(B: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    W = np.zeros(B.shape)
    for i, j in np.argwhere(B == 1):
        W[i, j] = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.5)
    return W


def _noise(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "gaussian":
        return rng.normal(size=n)
    if kind == "uniform":
        return rng.uniform(-np.sqrt(3), np.sqrt(3), size=n)
    if kind == "laplace":
        return rng.laplace(scale=1 / np.sqrt(2), size=n)
    raise ValueError(f"Unknown noise {kind!r}.")


def simulate_sem(
    B: np.ndarray,
    W: np.ndarray,
    n: int,
    noise: str,
    rng: np.random.Generator,
    quadratic_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Simulate an additive SEM with fixed weights.

    Edges with ``quadratic_mask[i, j]`` act through (z_i^2 - 1)/sqrt(2), where
    z_i is the standardized parent.  In the quadratic scenario every variable
    is standardized as soon as it is generated so that no variance explodes.
    """
    p = B.shape[0]
    X = np.zeros((n, p))
    for j in topological_order(B):
        signal = np.zeros(n)
        for i in np.flatnonzero(B[:, j]):
            x = X[:, i]
            if quadratic_mask is not None and quadratic_mask[i, j]:
                z = (x - x.mean()) / max(x.std(), 1e-12)
                x = (z ** 2 - 1.0) / np.sqrt(2.0)
            signal += W[i, j] * x
        X[:, j] = signal + _noise(noise, n, rng)
        if quadratic_mask is not None:
            X[:, j] = (X[:, j] - X[:, j].mean()) / max(X[:, j].std(), 1e-12)
    return X


def _hide(B: np.ndarray, n_latent: int, mode: str, rng: np.random.Generator):
    p = B.shape[0]
    if n_latent <= 0:
        return []
    if mode == "hubs":
        # Hidden common causes: the variables with the most children.
        out_degree = B.sum(axis=1) + rng.uniform(0, 0.1, size=p)
        return list(np.argsort(-out_degree)[:n_latent])
    return list(rng.choice(np.arange(p), size=n_latent, replace=False))


def make_scenario(
    name: str,
    p: int,
    n: int,
    n_test: int,
    rng: np.random.Generator,
    expected_degree: float = 2.0,
    noise: str = "gaussian",
    n_latent: int = 0,
    latent_mode: str = "random",
    quadratic_frac: float = 0.0,
) -> Scenario:
    """Draw a DAG over ``p`` variables (including hidden ones) and data from it."""
    B = generate_random_dag(p, expected_degree, rng)
    W = random_weights(B, rng)
    quad = None
    if name == "quadratic":
        quad = (rng.uniform(size=B.shape) < quadratic_frac) & (B == 1)
    elif name != "linear":
        raise ValueError(f"Unknown scenario {name!r}.")

    if quad is None:
        X = simulate_sem(B, W, n, noise, rng)
        X_test = simulate_sem(B, W, n_test, noise, rng) if n_test > 0 else np.zeros((0, p))
    else:
        # One draw, split afterwards, so that the per-draw standardization is
        # shared by the observed and the held-out sample (exchangeability).
        X_all = simulate_sem(B, W, n + n_test, noise, rng, quad)
        X, X_test = X_all[:n], X_all[n:]

    latent = set(int(v) for v in _hide(B, n_latent, latent_mode, rng))
    observed = [j for j in range(p) if j not in latent]
    names = [""] * p
    for k, j in enumerate(observed):
        names[j] = f"X{k}"
    for k, j in enumerate(sorted(latent)):
        names[j] = f"L{k}"
    cols = [names[j] for j in observed]
    data = pd.DataFrame(X[:, observed], columns=cols)
    test_data = pd.DataFrame(X_test[:, observed], columns=cols)
    full_dag = pd.DataFrame(B.astype(int), index=names, columns=names)
    true_graph = true_observed_graph(full_dag, cols)
    weights = W if (not latent and quad is None) else None
    return Scenario(data, test_data, full_dag, true_graph, weights, name)
