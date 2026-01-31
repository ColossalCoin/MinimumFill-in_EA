import os
import pickle
import pandas as pd
import networkx as nx
from scipy.sparse.csgraph import reverse_cuthill_mckee
from tqdm import tqdm

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