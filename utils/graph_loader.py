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
                # Ignorar comentarios, metadatos, líneas vacías,
                # comentarios ('c') y encabezados PACE ('p')
                line = line.strip()
                if not line or line.startswith(('c', 'p', '%', '#')):
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


# ----- SANITY CHECK -----
if __name__ == "__main__":
    print("Ejecutando prueba de carga de grafos...")

    # 1. Crear un archivo PACE sintético temporal
    test_filename = "test_graph_temp.gr"
    content = """c Comentario de prueba
    p cep 4 3
    1 2
    2 3
    3 4
    """
    with open(test_filename, 'w') as f:
        f.write(content)

    try:
        # 2. Cargar el grafo usando la función
        G = load_pace_graph(test_filename)

        # 3. Verificaciones (Asserts)
        # El grafo 1-2-3-4 es una línea (Path Graph) de 4 nodos y 3 aristas.
        assert G is not None, "La función retornó None"
        assert G.number_of_nodes() == 4, f"Nodos esperados 4, obtenidos {G.number_of_nodes()}"
        assert G.number_of_edges() == 3, f"Aristas esperadas 3, obtenidas {G.number_of_edges()}"

        # Verificar que el re-etiquetado funcionó (nodos deben ser 0, 1, 2, 3)
        assert sorted(list(G.nodes())) == [0, 1, 2, 3], "Los nodos no fueron re-etiquetados a enteros consecutivos"

        print("El cargador de grafos funciona correctamente")

    except AssertionError as e:
        print(f"Error en la prueba: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")
    finally:
        # 4. Limpieza: Borrar archivo temporal
        if os.path.exists(test_filename):
            os.remove(test_filename)