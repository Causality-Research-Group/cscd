from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from causallearn.utils.Dataset import load_dataset as causallearn_load_dataset
except Exception:
    causallearn_load_dataset = None

try:
    from pgmpy.example_models import load_model as pgmpy_load_model
    from pgmpy.sampling import BayesianModelSampling
except Exception:
    pgmpy_load_model = None
    BayesianModelSampling = None


def generate_random_dag(
    p: int,
    expected_degree: float,
    rng: np.random.Generator,
) -> np.ndarray:
    order = rng.permutation(p)
    prob = expected_degree / max(p - 1, 1)
    B = np.zeros((p, p), dtype=int)

    for a in range(p):
        for b in range(a + 1, p):
            if rng.uniform() < prob:
                parent = order[a]
                child = order[b]
                B[parent, child] = 1

    return B


def topological_order(B: np.ndarray) -> List[int]:
    p = B.shape[0]
    indeg = B.sum(axis=0).copy()
    remaining = set(range(p))
    order = []

    while remaining:
        roots = [j for j in remaining if indeg[j] == 0]
        if not roots:
            raise RuntimeError("Graph is cyclic.")
        j = roots[0]
        order.append(j)
        remaining.remove(j)
        for k in range(p):
            if B[j, k] == 1:
                indeg[k] -= 1

    return order


def simulate_linear_sem(
    B: np.ndarray,
    n: int,
    noise: str,
    rng: np.random.Generator,
    return_weights: bool = False,
):
    """Simulate data from a linear SEM with adjacency matrix B.

    Parameters
    ----------
    B : np.ndarray
        Binary adjacency matrix; B[i, j] == 1 means i -> j.
    n : int
        Number of samples.
    noise : str
        Noise distribution: 'gaussian', 'uniform', or 'laplace'.
    rng : np.random.Generator
        Random number generator.
    return_weights : bool, optional
        If True, return (data, W) where W[i, j] is the linear coefficient
        for the edge i -> j.  Default False preserves backward compatibility.

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, np.ndarray)
        Simulated data frame, and optionally the weight matrix W.
    """
    p = B.shape[0]
    W = np.zeros((p, p))

    for i, j in np.argwhere(B == 1):
        W[i, j] = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.5)

    X = np.zeros((n, p))
    order = topological_order(B)

    for j in order:
        parents = np.where(B[:, j] == 1)[0]
        signal = X[:, parents] @ W[parents, j] if len(parents) else 0.0

        if noise == "gaussian":
            eps = rng.normal(size=n)
        elif noise == "uniform":
            eps = rng.uniform(-np.sqrt(3), np.sqrt(3), size=n)
        elif noise == "laplace":
            eps = rng.laplace(scale=1 / np.sqrt(2), size=n)
        else:
            raise ValueError(f"Unknown noise: {noise}")

        X[:, j] = signal + eps

    data = pd.DataFrame(X, columns=[f"X{k}" for k in range(p)])
    if return_weights:
        return data, W
    return data


def hide_latents(
    data: pd.DataFrame,
    B: np.ndarray,
    latent_fraction: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hide a random fraction of the variables.

    Returns the observed data, with columns renamed X0, X1, ..., and the full
    data-generating DAG as a labelled adjacency matrix.  In that matrix the
    observed variables carry their new names and hidden variables are named
    L0, L1, ....  The truth over the observed variables is the *marginal* of
    this DAG, not its induced subgraph (see ``true_observed_graph``).
    """
    p = B.shape[0]
    n_latent = int(np.floor(latent_fraction * p))
    latent = set(rng.choice(np.arange(p), size=n_latent, replace=False)) if n_latent > 0 else set()
    observed = [j for j in range(p) if j not in latent]

    names = [""] * p
    for k, j in enumerate(observed):
        names[j] = f"X{k}"
    for k, j in enumerate(sorted(latent)):
        names[j] = f"L{k}"

    data_obs = data.iloc[:, observed].copy()
    data_obs.columns = [names[j] for j in observed]
    full_dag = pd.DataFrame(B.astype(int), index=names, columns=names)
    return data_obs, full_dag


def true_observed_graph(full_dag: pd.DataFrame, observed: List[str]) -> pd.DataFrame:
    """True parent claims among the observed variables.

    Entry [a, b] is 1 iff a -> b in the MAG obtained by marginalizing out the
    hidden variables: a and b cannot be separated by observed variables, and
    a is an ancestor of b.  With no hidden variables this is the DAG itself.
    """
    from .mixed_graph import dag_from_adjacency, dag_marginal_mag, directed_adjacency

    mag = dag_marginal_mag(dag_from_adjacency(full_dag), observed)
    return directed_adjacency(mag)


def load_causallearn_sachs() -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if causallearn_load_dataset is None:
        raise ImportError(
            "Could not import causallearn.utils.Dataset.load_dataset. "
            "Please install causal-learn."
        )

    data_np, labels = causallearn_load_dataset("sachs")
    data = pd.DataFrame(data_np, columns=list(labels))
    reference_graph = sachs_reference_graph(list(data.columns))
    return data, reference_graph


def pgmpy_model_to_adjacency(model, var_names: Optional[List[str]] = None) -> pd.DataFrame:
    if var_names is None:
        var_names = list(model.nodes())

    A = pd.DataFrame(0, index=var_names, columns=var_names, dtype=int)
    for u, v in model.edges():
        if u in A.index and v in A.columns:
            A.loc[u, v] = 1
    return A


def encode_discrete_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            out[col] = df[col]
        else:
            out[col] = pd.Categorical(df[col]).codes
    return out.astype(float)


BNLEARN_NETWORKS = ("asia", "sachs", "child", "insurance", "alarm")


def load_bnlearn_pgmpy_dataset(
    name: str,
    n_samples: int,
    random_seed: int = 123,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if pgmpy_load_model is None or BayesianModelSampling is None:
        raise ImportError(
            "Could not import pgmpy modules. Please install pgmpy."
        )

    name = name.lower()
    if name not in BNLEARN_NETWORKS:
        raise ValueError(f"Expected one of: {', '.join(BNLEARN_NETWORKS)}.")

    model = pgmpy_load_model(f"bnlearn/{name}")
    var_names = list(model.nodes())
    sampler = BayesianModelSampling(model)

    try:
        sampled = sampler.forward_sample(
            size=n_samples,
            seed=random_seed,
            show_progress=False,
        )
    except TypeError:
        sampled = sampler.forward_sample(
            size=n_samples,
            seed=random_seed,
        )

    data = encode_discrete_dataframe(sampled[var_names])
    true_graph = pgmpy_model_to_adjacency(model, var_names)
    return data, true_graph


def load_real_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.select_dtypes(include=[np.number]).copy()
    df = df.fillna(df.mean(numeric_only=True))
    return df


# Consensus network of Sachs et al. (2005), as distributed with bnlearn (17 arcs).
SACHS_CONSENSUS_EDGES = [
    ("Erk", "Akt"), ("Mek", "Erk"), ("PIP3", "PIP2"), ("PKA", "Akt"), ("PKA", "Erk"),
    ("PKA", "Jnk"), ("PKA", "Mek"), ("PKA", "P38"), ("PKA", "Raf"), ("PKC", "Jnk"),
    ("PKC", "Mek"), ("PKC", "P38"), ("PKC", "PKA"), ("PKC", "Raf"), ("Plcg", "PIP2"),
    ("Plcg", "PIP3"), ("Raf", "Mek"),
]

# causal-learn names the variables in lower case and calls Plcg "plc".
_SACHS_ALIASES = {"plc": "plcg"}


def sachs_reference_graph(var_names: List[str]) -> Optional[pd.DataFrame]:
    """Consensus Sachs network over ``var_names``, matching names case-insensitively."""
    canonical = {_SACHS_ALIASES.get(v.lower(), v.lower()): v for v in var_names}
    A = pd.DataFrame(0, index=var_names, columns=var_names, dtype=int)
    n_found = 0
    for a, b in SACHS_CONSENSUS_EDGES:
        a_name, b_name = canonical.get(a.lower()), canonical.get(b.lower())
        if a_name is not None and b_name is not None:
            A.loc[a_name, b_name] = 1
            n_found += 1
    return A if n_found else None
