from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DAG:
    """Directed acyclic graph with topologically ordered node ids."""

    n_nodes: int
    edges: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.n_nodes <= 1:
            raise ValueError("n_nodes must be greater than one")
        if not self.edges:
            raise ValueError("edges must not be empty")
        if any(u < 0 or v >= self.n_nodes or u >= v for u, v in self.edges):
            raise ValueError("edges must satisfy 0 <= u < v < n_nodes")

    def path_vector(self, path: tuple[int, ...]) -> np.ndarray:
        """Return the binary edge-incidence vector for a path."""
        if len(path) < 2:
            raise ValueError("path must contain at least two nodes")
        edge_index = {edge: index for index, edge in enumerate(self.edges)}
        vector = np.zeros(len(self.edges), dtype=float)
        for edge in zip(path[:-1], path[1:]):
            if edge not in edge_index:
                raise ValueError(f"path uses missing edge {edge}")
            vector[edge_index[edge]] = 1.0
        return vector

    def enumerate_paths(self, source: int, target: int) -> list[tuple[int, ...]]:
        """Enumerate every source-target path in the DAG."""
        if not 0 <= source < target < self.n_nodes:
            raise ValueError("expected 0 <= source < target < n_nodes")

        outgoing: dict[int, list[int]] = {node: [] for node in range(self.n_nodes)}
        for u, v in self.edges:
            outgoing[u].append(v)

        paths: list[tuple[int, ...]] = []

        def dfs(node: int, path: tuple[int, ...]) -> None:
            if node == target:
                paths.append(path)
                return
            for nxt in outgoing[node]:
                if nxt <= target:
                    dfs(nxt, path + (nxt,))

        dfs(source, (source,))
        if not paths:
            raise ValueError(f"no path exists from {source} to {target}")
        return paths

    def shortest_path(self, costs: np.ndarray, source: int, target: int) -> tuple[int, ...]:
        """Return a minimum-cost source-target path."""
        costs = np.asarray(costs, dtype=float)
        if costs.shape != (len(self.edges),):
            raise ValueError("costs must contain one value per edge")
        paths = self.enumerate_paths(source, target)
        return min(paths, key=lambda path: float(self.path_vector(path) @ costs))
