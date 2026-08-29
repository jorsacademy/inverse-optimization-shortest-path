from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .graph import DAG


def recover_edge_costs(
    graph: DAG,
    demonstrations: list[tuple[int, int, tuple[int, ...]]],
    prior: np.ndarray | None = None,
    margin: float = 0.05,
) -> np.ndarray:
    """Recover positive edge costs that rationalize demonstrated shortest paths.

    Scale is identified by enforcing ``sum(costs) == number_of_edges``. The inverse
    problem minimizes L1 deviation from a prior while imposing max-margin preference
    inequalities against every alternative path for each demonstrated OD pair.
    """
    if not demonstrations:
        raise ValueError("demonstrations must not be empty")
    if margin < 0:
        raise ValueError("margin must be nonnegative")

    n_edges = len(graph.edges)
    if prior is None:
        prior = np.ones(n_edges, dtype=float)
    prior = np.asarray(prior, dtype=float)
    if prior.shape != (n_edges,):
        raise ValueError("prior must contain one value per edge")

    objective = np.r_[np.zeros(n_edges), np.ones(n_edges)]
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for source, target, demonstrated_path in demonstrations:
        demonstrated = graph.path_vector(demonstrated_path)
        for alternative_path in graph.enumerate_paths(source, target):
            if alternative_path == demonstrated_path:
                continue
            alternative = graph.path_vector(alternative_path)
            # c^T x_demo + margin <= c^T x_alt
            rows.append(np.r_[demonstrated - alternative, np.zeros(n_edges)])
            rhs.append(-margin)

    # Linearize |c - prior| with nonnegative deviation variables d.
    for index in range(n_edges):
        positive = np.zeros(2 * n_edges)
        positive[index] = 1.0
        positive[n_edges + index] = -1.0
        rows.append(positive)
        rhs.append(float(prior[index]))

        negative = np.zeros(2 * n_edges)
        negative[index] = -1.0
        negative[n_edges + index] = -1.0
        rows.append(negative)
        rhs.append(float(-prior[index]))

    equality = np.zeros((1, 2 * n_edges))
    equality[0, :n_edges] = 1.0
    bounds = [(1e-3, None)] * n_edges + [(0.0, None)] * n_edges

    result = linprog(
        objective,
        A_ub=np.asarray(rows),
        b_ub=np.asarray(rhs),
        A_eq=equality,
        b_eq=np.asarray([float(n_edges)]),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"inverse LP failed: {result.message}")
    return result.x[:n_edges]
