import numpy as np
import pytest

from inverse_opt import DAG, recover_edge_costs


def test_recovered_costs_rationalize_demonstrations() -> None:
    graph = DAG(5, ((0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (3, 4)))
    true_costs = np.array([0.7, 1.4, 0.8, 0.5, 2.0, 0.6])
    od_pairs = [(0, 4), (0, 3), (1, 4)]
    demonstrations = [
        (source, target, graph.shortest_path(true_costs, source, target))
        for source, target in od_pairs
    ]

    recovered = recover_edge_costs(graph, demonstrations)

    assert np.all(recovered > 0)
    assert np.isclose(recovered.sum(), len(graph.edges))
    for source, target, path in demonstrations:
        assert graph.shortest_path(recovered, source, target) == path


def test_inverse_problem_validates_inputs() -> None:
    graph = DAG(3, ((0, 1), (1, 2), (0, 2)))
    with pytest.raises(ValueError, match="demonstrations"):
        recover_edge_costs(graph, [])
    with pytest.raises(ValueError, match="margin"):
        recover_edge_costs(graph, [(0, 2, (0, 2))], margin=-0.1)
