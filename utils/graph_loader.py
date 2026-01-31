import networkx as nx
import os


def load_pace_graph(filepath):
    """
    Carga un grafo desde un archivo formato PACE (.graph) a NetworkX.
    Realiza la limpieza y estandarización de etiquetas.

    :param filepath: Ruta al archivo.
    :return: Grafo nx.Graph con nodos enteros 0,..., N-1, o None si falla.
    """
    G = nx.Graph()
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el archivo: {filepath}")

        with open(filepath, 'r') as f:
            for line in f:
                # Ignorar comentarios y metadatos
                if line.startswith(('#', '%')):
                    continue

                parts = line.split()

                # Leer aristas (formato: u v)
                if len(parts) >= 2:
                    u, v = parts[0], parts[1]
                    G.add_edge(u, v)

        # ESTANDARIZACIÓN:
        # Convertir etiquetas a enteros consecutivos (0, 1, 2, ..., N-1)
        return nx.convert_node_labels_to_integers(G)

    except Exception as e:
        print(f"Error cargando {filepath}: {e}")
        return None