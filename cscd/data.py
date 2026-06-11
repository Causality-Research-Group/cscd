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
) -> pd.DataFrame:
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

    return pd.DataFrame(X, columns=[f"X{k}" for k in range(p)])


def hide_latents(
    data: pd.DataFrame,
    B: np.ndarray,
    latent_fraction: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, np.ndarray]:
    p = B.shape[0]
    n_latent = int(np.floor(latent_fraction * p))

    if n_latent <= 0:
        return data, B

    latent = set(rng.choice(np.arange(p), size=n_latent, replace=False))
    observed = [j for j in range(p) if j not in latent]

    data_obs = data.iloc[:, observed].copy()
    B_obs = B[np.ix_(observed, observed)]

    data_obs.columns = [f"X{k}" for k in range(len(observed))]
    return data_obs, B_obs


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
    if name not in {"asia", "sachs", "child"}:
        raise ValueError("Expected one of: asia, sachs, child.")

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


def sachs_reference_graph(var_names: List[str]) -> Optional[pd.DataFrame]:
    edges = [
        ("Raf", "Mek"),
        ("Mek", "Erk"),
        ("Plcg", "PIP2"),
        ("Plcg", "PIP3"),
        ("PIP3", "Akt"),
        ("PIP3", "PIP2"),
        ("PKA", "Raf"),
        ("PKA", "Mek"),
        ("PKA", "Erk"),
        ("PKC", "Raf"),
        ("PKC", "Mek"),
        ("PKC", "Jnk"),
        ("PKC", "P38"),
    ]

    var_set = set(var_names)
    usable = [(a, b) for a, b in edges if a in var_set and b in var_set]

    if not usable:
        return None

    A = pd.DataFrame(0, index=var_names, columns=var_names, dtype=int)
    for a, b in usable:
        A.loc[a, b] = 1
    return A
