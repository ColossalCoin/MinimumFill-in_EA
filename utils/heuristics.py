import networkx as nx

DATASET_DIR = "datasets"
BASELINE_FILE = "baselines.csv"


def count_fillin(G, ordering):
    """
    Simula la eliminación gaussiana y cuenta las aristas de relleno (fill-in).

    :param G: Grafo NetworkX original.
    :param ordering: Lista de nodos en el orden de eliminación.
    :return: int (Número total de aristas añadidas)
    """

    # Trabajamos sobre una copia para no destruir el original
    H = G.copy()
    fill_in_count = 0

    # Simulación del proceso de eliminación
    for node in ordering:
        neighbors = list(H.neighbors(node))

        # Conectar todos los vecinos entre sí (formar un clique)
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                u, v = neighbors[i], neighbors[j]

                # Si no están conectados, agregamos arista (Fill-in)
                if not H.has_edge(u, v):
                    H.add_edge(u, v)
                    fill_in_count += 1

        # Eliminamos el nodo procesado (en la práctica, NetworkX lo quita del grafo)
        H.remove_node(node)

    return fill_in_count


def greedy_minimum_degree(G, compute_cost=False):
    """
    Implementación del algoritmo de mínimo grado (MD).

    Args:
    :param G: Grafo de NetworkX.
    :param compute_cost (bool): Si es True, devuelve una tupla (orden, costo_fillin).
                                Si es False, devuelve solo la lista del orden (comportamiento original).

    :return list o (list, int): Orden de eliminación y opcionalmente el costo.
    """
    # Trabajamos sobre una copia para no destruir el grafo original
    H = G.copy()
    ordering = []

    while H.number_of_nodes() > 0:
        # 1. Encontrar nodo con grado mínimo
        # Usamos el grado y el ID del nodo como criterio de desempate
        degrees = dict(H.degree())
        min_node = min(degrees, key=lambda k: (degrees[k], k))

        # 2. Simular Fill-in local (conectar vecinos)
        neighbors = list(H.neighbors(min_node))

        # Iteramos sobre los pares de vecinos para formar el clique
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                u, v = neighbors[i], neighbors[j]
                if not H.has_edge(u, v):
                    H.add_edge(u, v)

        # 3. Eliminar nodo y guardar en orden
        H.remove_node(min_node)
        ordering.append(min_node)

    if compute_cost:
        fill_in_count = count_fillin(G, ordering)
        return ordering, fill_in_count

    return ordering


# ----- SANITY CHECK -----
if __name__ == "__main__":
    print("Ejecutando pruebas unitarias internas...")

    # CASO DE PRUEBA: Grafo Lineal P5 (0-1-2-3-4)
    # Un árbol o línea no necesita aristas de relleno si se elimina desde las hojas.
    # El algoritmo AMD detecta hojas (grado 1) y las elimina primero.

    G_test = nx.path_graph(5)

    # 1. Prueba manual: Orden malo
    manual_order = [3, 2, 1, 0, 4]
    manual_cost = count_fillin(G_test, manual_order)
    print(f"1. Orden manual [3, 2, 1, 0, 4] -> Costo: {manual_cost} (Esperado: 3)")

    # 2. Prueba AMD: Debería encontrar el óptimo (0)
    amd_order, amd_cost = greedy_minimum_degree(G_test, compute_cost=True)
    print(f"2. Orden AMD {amd_order} -> Costo: {amd_cost} (Esperado: 0)")

    # Validaciones automáticas
    try:
        assert manual_cost == 3, "Error en count_fillin (falso negativo)"
        assert amd_cost == 0, "Error en greedy (no encontró óptimo) o bug de conteo"
        print("Todas las pruebas pasaron exitosamente")
    except AssertionError as e:
        print(e)