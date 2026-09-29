from __future__ import annotations

import numpy as np

DATASET_DIR = "datasets"
BASELINE_FILE = "baselines.csv"


def count_fillin(adj_matrix: np.ndarray, ordering, *, validate: bool = True) -> int:
    """
    Simula la eliminación gaussiana y cuenta las aristas de relleno (fill-in).

    :param adj_matrix: Matriz de adyacencia (numpy.ndarray).
    :param ordering: Lista de nodos en el orden de eliminación.
    :return: int (Número total de aristas añadidas)
    """

    A = np.asarray(adj_matrix)
    if validate:
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("adj_matrix debe ser una matriz cuadrada (n x n).")
        n = int(A.shape[0])
        H = (A != 0).copy()
        if not np.array_equal(H, H.T):
            H |= H.T
        H[np.arange(n), np.arange(n)] = False
    else:
        # Camino rápido: asumimos (n x n), simétrica, diagonal en 0, ya normalizada.
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("adj_matrix debe ser una matriz cuadrada (n x n).")
        n = int(A.shape[0])
        H = A.copy()
        if H.dtype != bool:
            H = (H != 0)

    active = np.ones(n, dtype=bool)
    fill_in_count = 0

    for node in ordering:
        if not active[node]:
            continue

        neigh_mask = H[node] & active
        neigh_mask[node] = False
        neighbors = np.nonzero(neigh_mask)[0]
        k = int(neighbors.size)

        if k > 1:
            sub = H[np.ix_(neighbors, neighbors)]
            existing = int(sub.sum() // 2)  # simétrico, sin diagonal
            total = k * (k - 1) // 2
            fill_in_count += total - existing

            # Formar clique entre vecinos
            H[np.ix_(neighbors, neighbors)] = True
            H[neighbors, neighbors] = False  # sin autolazos en el sub-bloque

        # Eliminar nodo
        H[node, :] = False
        H[:, node] = False
        active[node] = False

    return int(fill_in_count)


def greedy_minimum_degree(G, compute_cost=False):
    """
    Implementación del algoritmo de mínimo grado (MD).

    Args:
    :param G: Matriz de adyacencia (numpy.ndarray).
    :param compute_cost (bool): Si es True, devuelve una tupla (orden, costo_fillin).
                                Si es False, devuelve solo la lista del orden (comportamiento original).

    :return list o (list, int): Orden de eliminación y opcionalmente el costo.
    """
    A = np.asarray(G)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("G debe ser una matriz cuadrada (n x n).")
    n = int(A.shape[0])
    H = (A != 0).copy()
    if not np.array_equal(H, H.T):
        H |= H.T
    H[np.arange(n), np.arange(n)] = False

    active = np.ones(n, dtype=bool)
    ordering: list[int] = []

    while active.any():
        active_idx = np.nonzero(active)[0]
        # grado dentro del subgrafo activo
        degrees = H[np.ix_(active_idx, active_idx)].sum(axis=1)
        min_pos = int(np.argmin(degrees))
        min_node = int(active_idx[min_pos])

        neigh_mask = H[min_node] & active
        neigh_mask[min_node] = False
        neighbors = np.nonzero(neigh_mask)[0]
        if neighbors.size > 1:
            H[np.ix_(neighbors, neighbors)] = True
            H[neighbors, neighbors] = False

        H[min_node, :] = False
        H[:, min_node] = False
        active[min_node] = False
        ordering.append(min_node)

    if compute_cost:
        return ordering, count_fillin(A, ordering)
    return ordering, None


# ----- SANITY CHECK -----
if __name__ == "__main__":
    print("Ejecutando pruebas unitarias internas...")

    # CASO DE PRUEBA: Grafo Lineal P5 (0-1-2-3-4)
    # Un árbol o línea no necesita aristas de relleno si se elimina desde las hojas.
    # El algoritmo AMD detecta hojas (grado 1) y las elimina primero.

    # Grafo lineal P5 en matriz de adyacencia
    n = 5
    G_test = np.zeros((n, n), dtype=np.uint8)
    for i in range(n - 1):
        G_test[i, i + 1] = 1
        G_test[i + 1, i] = 1

    # 1. Prueba manual: Orden malo
    manual_order = [3, 2, 1, 0, 4]
    manual_cost = count_fillin(G_test, manual_order)
    print(f"1. Orden manual [3, 2, 1, 0, 4] -> Costo: {manual_cost} (Esperado: 3)")

    # 2. Prueba MD: Debería encontrar el óptimo (0)
    amd_order, amd_cost = greedy_minimum_degree(G_test, compute_cost=True)
    print(f"2. Orden AMD {amd_order} -> Costo: {amd_cost} (Esperado: 0)")

    # Validaciones automáticas
    try:
        assert manual_cost == 3, "Error en count_fillin (falso negativo)"
        assert amd_cost == 0, "Error en greedy (no encontró óptimo) o bug de conteo"
        print("Todas las pruebas pasaron exitosamente")
    except AssertionError as e:
        print(e)