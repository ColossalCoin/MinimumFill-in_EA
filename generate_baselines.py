import os
import pickle
import pandas as pd
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from tqdm import tqdm

DATASET_DIR = "datasets"
BASELINE_FILE = "baselines.csv"


def count_fillin(graph, ordering=None):
    """
    Simula la eliminación gaussiana y cuenta las aristas de relleno (fill-in).

    :param graph: Grafo NetworkX original.
    :param ordering: Lista de nodos en el orden de eliminación.
                     Si es None, usa el orden natural (0, 1, 2...).
    :return: int (Número total de aristas añadidas)
    """
    # Trabajamos sobre una copia para no destruir el original
    H = graph.copy()

    # Si hay un orden específico, re-etiquetamos los nodos para procesarlos en orden 0, 1, 2...
    if ordering is not None:
        # Mapeo: Nodo -> Posición en el orden
        mapping = {node: i for i, node in enumerate(ordering)}
        H = nx.relabel_nodes(H, mapping)
    else:
        # Aseguramos que los nodos sean enteros consecutivos si es orden natural
        H = nx.convert_node_labels_to_integers(H)

    fill_in_count = 0
    nodes = sorted(list(H.nodes()))  # Procesamos 0, 1, 2...

    # Simulación del proceso de eliminación
    for node in nodes:
        neighbors = list(H.neighbors(node))

        # Conectar todos los vecinos entre sí (formar un clique)
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                u, v = neighbors[i], neighbors[j]
                if not H.has_edge(u, v):
                    H.add_edge(u, v)
                    fill_in_count += 1

        # Eliminamos el nodo procesado (en la práctica, NetworkX lo quita del grafo)
        H.remove_node(node)

    return fill_in_count


def greedy_minimum_degree(graph):
    """
    Implementación del algoritmo de mínimo grado aproximado (AMD).
    Selecciona siempre el nodo con menor grado actual.

    :return: Lista con el orden de eliminación.
    """
    H = graph.copy()
    ordering = []

    # Usamos una barra de progreso manual pequeña o silencio
    # Es una heurística constructiva voraz
    while H.number_of_nodes() > 0:
        # 1. Encontrar nodo con grado mínimo
        # (Ordenamos por grado y luego por ID para determinismo en empates)
        degrees = dict(H.degree())
        min_node = min(degrees, key=lambda k: (degrees[k], k))

        # 2. Simular Fill-in local (conectar vecinos)
        neighbors = list(H.neighbors(min_node))
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                u, v = neighbors[i], neighbors[j]
                if not H.has_edge(u, v):
                    H.add_edge(u, v)

        # 3. Eliminar nodo y guardar en orden
        H.remove_node(min_node)
        ordering.append(min_node)

    return ordering


def run_benchmarks():
    results = []

    # 1. Encontrar todos los grafos .pkl
    graph_files = []
    for root, dirs, filenames in os.walk(DATASET_DIR):
        for filename in filenames:
            if filename.endswith(".pkl"):
                graph_files.append(os.path.join(root, filename))

    print(f"--- Generando Línea Base para {len(graph_files)} grafos ---")
    print("Calculando: Natural, RCM (SciPy) y Minimum Degree (Greedy)...")

    # 2. Iterar y procesar
    for filepath in tqdm(graph_files):
        # Cargar Grafo
        with open(filepath, 'rb') as f:
            G = pickle.load(f)

        category = os.path.basename(os.path.dirname(filepath))
        name = os.path.basename(filepath).replace(".pkl", "")

        # A. Fill-in Orden Natural (Original)
        fi_natural = count_fillin(G, ordering=None)

        # B. Fill-in RCM (Reverse Cuthill-McKee)
        # Convertimos a matriz dispersa SciPy para usar su algoritmo optimizado
        adj_matrix = nx.to_scipy_sparse_array(G, format='csr')
        perm_rcm = reverse_cuthill_mckee(adj_matrix, symmetric_mode=True)
        # La función de scipy devuelve la permutación, hay que mapearla a los nodos
        nodes_list = list(G.nodes())  # Asumiendo que están ordenados 0..N-1
        ordering_rcm = [nodes_list[i] for i in perm_rcm]
        fi_rcm = count_fillin(G, ordering_rcm)

        # C. Fill-in Minimum Degree (Heurística Constructiva)
        ordering_md = greedy_minimum_degree(G)
        fi_md = count_fillin(G, ordering_md)

        # Guardar resultados
        results.append({
            "Category": category,
            "Graph": name,
            "Nodes": G.number_of_nodes(),
            "Edges": G.number_of_edges(),
            "Base_Natural": fi_natural,
            "Base_RCM": fi_rcm,
            "Base_MinDegree": fi_md
        })

    # 3. Exportar CSV
    df = pd.DataFrame(results)
    df.to_csv(BASELINE_FILE, index=False)
    print(f"\n✅ Benchmarks guardados en '{BASELINE_FILE}'")

    # Mostrar resumen rápido en consola
    print("\nResumen Promedio de Fill-in:")
    print(df.groupby("Category")[["Base_Natural", "Base_RCM", "Base_MinDegree"]].mean())


if __name__ == "__main__":
    run_benchmarks()