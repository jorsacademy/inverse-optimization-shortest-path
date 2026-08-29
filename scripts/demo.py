import numpy as np

from inverse_opt import DAG, recover_edge_costs


graph = DAG(5, ((0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (3, 4)))
true_costs = np.array([0.7, 1.4, 0.8, 0.5, 2.0, 0.6])
od_pairs = [(0, 4), (0, 3), (1, 4)]
demonstrations = [
    (source, target, graph.shortest_path(true_costs, source, target))
    for source, target in od_pairs
]
recovered = recover_edge_costs(graph, demonstrations)

print("true      ", np.round(true_costs, 3))
print("recovered ", np.round(recovered, 3))
for source, target, path in demonstrations:
    print(
        (source, target),
        "demo=",
        path,
        "recovered_opt=",
        graph.shortest_path(recovered, source, target),
    )
