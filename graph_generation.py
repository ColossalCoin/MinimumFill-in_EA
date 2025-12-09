import networkx as nx
import os
import pickle
import matplotlib.pyplot as plt
from scipy.io import mmread

OUTPUT_DIR = "datasets"


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_graph(graph, name, category, instance_id=None):
    """
    Guarda un grafo serializándolo en un archivo pickle y generando una representación visual del grafo.
    Si se proporciona un instance_id, se añade al nombre del archivo para diferenciar réplicas.

    :param graph: El grafo a guardar.
    :type graph: nx.Graph
    :param name: El nombre base del archivo (sin extensión).
    :type name: str
    :param category: La carpeta de categoría donde se guardarán el grafo y su visualización.
    :type category: str
    :param instance_id: Identificador de la instancia (semilla) para evitar sobreescritura. Si es None, se ignora.
    :type instance_id: int, optional
    :return: None
    """
    path = os.path.join(OUTPUT_DIR, category)
    ensure_dir(path)

    # Construcción del nombre: Si hay ID de instancia, lo agregamos al nombre
    # Ejemplo: Barabasi_N100_m2_I0.pkl
    if instance_id is not None:
        filename_base = f"{name}_I{instance_id}"
    else:
        filename_base = name

    # Guardamos el grafo en un archivo pickle
    filename_pkl = os.path.join(path, f"{filename_base}.pkl")
    with open(filename_pkl, 'wb') as file:
        pickle.dump(graph, file)

    # Generamos imagen solo para la primera instancia (I0) o si es único para ahorrar espacio
    if instance_id is None or instance_id == 0:
        plt.figure(figsize=(6, 6))

        # Aplicamos layouts específicos según el tipo de visualización
        if "Grid" in category:
            # Spring layout suele visualizar mejor la estructura general si no tenemos coordenadas
            pos = nx.spring_layout(graph, seed=42)
        elif "Watts" in category:
            pos = nx.circular_layout(graph)
        else:
            pos = nx.spring_layout(graph, seed=42)

        nx.draw(graph, pos, node_size=20, node_color='blue', alpha=0.6, width=0.5)
        plt.title(f"{filename_base} (N={graph.number_of_nodes()}, E={graph.number_of_edges()})")
        plt.savefig(os.path.join(path, f"{filename_base}.png"))
        plt.close()

    print(f"Generado: {filename_base} en {category}")


def generate_synthetic_grids(grids: list[tuple[int, int]]):
    """
    Genera y guarda grafos sintéticos con estructura de malla basándose en las dimensiones especificadas.
    Las mallas son deterministas, por lo que se generan una sola vez.

    :param grids: Lista de tuplas que especifican las dimensiones (altura, ancho) de los grafos de cuadrícula a generar.
    :type grids: list[tuple[int, int]]
    :return: None
    """
    for h, w in grids:
        graph = nx.grid_2d_graph(h, w)
        # Convertimos etiquetas de tuplas (0,1) a enteros (1, 2...) para DEAP
        graph = nx.convert_node_labels_to_integers(graph)
        name = f"Grid_{h}x{w}_N{h * w}"
        save_graph(graph, name, "synthetic_grid")


def generate_synthetic_barabasi(sizes: list[int], m: int, SEED: int, instance_id: int):
    """
    Genera grafos sintéticos de Barabási-Albert variando la semilla para crear múltiples instancias.

    :param sizes: Una lista de números enteros que indica los tamaños (número de nodos) de los respectivos grafos.
    :type sizes: list[int]
    :param m: Un número entero que especifica el número de aristas que cada nuevo nodo añade al grafo.
    :type m: int
    :param SEED: La semilla específica para esta instancia.
    :type SEED: int
    :param instance_id: Identificador numérico de la instancia actual (0, 1, 2...).
    :type instance_id: int
    :return: None
    """
    for n in sizes:
        graph = nx.barabasi_albert_graph(n, m=m, seed=SEED)
        name = f"Barabasi_N{n}_m{m}"
        save_graph(graph, name, "synthetic_barabasi", instance_id)


def generate_synthetic_watts(sizes: list[int], k: int, p: float, SEED: int, instance_id: int):
    """
    Genera grafos sintéticos basándose en el modelo de mundo pequeño (small-world) de Watts-Strogatz.

    :param sizes: Lista de números enteros que representan los tamaños de los grafos (número de nodos).
    :type sizes: list[int]
    :param k: El número de vecinos más cercanos en la topología de anillo inicial.
    :type k: int
    :param p: La probabilidad de reconexión (rewiring) para las aristas en el grafo.
    :type p: float
    :param SEED: La semilla específica para esta instancia.
    :type SEED: int
    :param instance_id: Identificador numérico de la instancia actual.
    :type instance_id: int
    :return: None
    """
    for n in sizes:
        graph = nx.watts_strogatz_graph(n, k=k, p=p, seed=SEED)
        name = f"WattsStrogatz_N{n}_k{k}_p{str(p).replace('.', '')}"
        save_graph(graph, name, "synthetic_smallworld", instance_id)


def fetch_real_benchmarks():
    """
    Procesa las matrices reales de la colección SuiteSparse.
    Busca archivos .mtx en la carpeta local 'SuiteSparse', los convierte a grafos y los serializa.

    :return: None
    """
    # Tu lista completa original
    real_matrices = [
        "494_bus", "bcsstk02", "bcsstk03", "bcsstk04", "bcsstk05",
        "bcsstk06", "bcsstk07", "bcsstk20", "bcsstk22", "bcsstm02",
        "bcsstm05", "bcsstm06", "bcsstm07", "bcsstm20", "bcsstm22",
        "Journals", "lund_a", "lund_b", "mesh2e1", "mesh2em5",
        "mesh3e1", "mesh3em5", "mhdb416", "nos1", "nos4", "nos5",
        "plat362", "Trefethen_150", "Trefethen_200", "Trefethen_200b",
        "Trefethen_300", "Trefethen_500"
    ]
    base_path = "SuiteSparse"  # Asegúrate que esta carpeta exista y tenga los .mtx
    ensure_dir(base_path)

    print(f"--- Procesando {len(real_matrices)} matrices reales ---")

    for mtx_file in real_matrices:
        try:
            full_path = os.path.join(base_path, mtx_file + '.mtx')

            if not os.path.exists(full_path):
                print(f"Saltando {mtx_file}: Archivo no encontrado en {base_path}")
                continue

            matrix = mmread(full_path)

            # Convertimos a grafo de NetworkX
            graph = nx.from_scipy_sparse_array(matrix)

            # Quitamos bucles (aristas de un nodo a sí mismo) si existen
            graph.remove_edges_from(nx.selfloop_edges(graph))

            # Guardar como benchmark (Id None porque son únicos)
            save_graph(graph, mtx_file, "Real_SuiteSparse")

        except Exception as e:
            print(f"Error procesando {mtx_file}: {e}")


if __name__ == "__main__":
    SEEDS = [42, 101, 666, 1234, 5678]

    print("--- Iniciando generación de Datasets ---")

    # 1. Generar Sintéticos Estocásticos (5 instancias diferentes por configuración)
    for i, seed in enumerate(SEEDS):
        print(f"Generando instancia {i} (Semilla: {seed})...")
        generate_synthetic_barabasi(sizes=[100, 200, 300], m=2, SEED=seed, instance_id=i)
        generate_synthetic_watts(sizes=[100, 200, 300], k=4, p=0.1, SEED=seed, instance_id=i)

    # 2. Generar Grids (Deterministas, solo se corren una vez fuera del bucle de semillas)
    generate_synthetic_grids(grids=[(8, 8), (12, 12), (15, 15)])

    # 3. Procesar Reales (Solo se corren una vez)
    fetch_real_benchmarks()

    print("\nTodos los grafos generados en la carpeta 'datasets/'")