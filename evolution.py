import random
import networkx as nx
from deap import creator, base, tools

# DEAP requiere que se indique si la función de aptitud debe ser minimizada o maximizada. Declaramos que la función
# de aptitud debe ser minimizada usando weights=(-1.0,)
creator.create(name="FitnessMin", base.Fitness, weights=(-1.0,))

# Creamos un individuo de tipo permutation a partir de una lista. Usar arreglos de numpy no es conveniente en este caso
# dado que DEAP utiliza Python puro, es decir, requeriría convertir los arreglos en listas cada iteración.
creator.create(name="Individual", list, fitness=creator.FitnessMin)


# Definimos el AE como una clase.
class GraphChordalizer:
    def __init__(self, matrix_data):
        """
        Constructor para inicializar una instancia con atributos de datos específicos.

        Inicializa la representación del grafo a partir de la matriz proporcionada usando NetworkX, calcula el número
        de nodos en el grafo y configura una instancia de caja de herramientas (toolbox) para operaciones posteriores.

        :param matrix_data: Matriz de adyacencia cuadrada que define la estructura del grafo. Cada entrada en la matriz
            representa si los vértices correspondientes osn adyacentes.
        :type matrix_data: numpy.ndarray
        """
        # Suponemos que la matriz del constructor es un arreglo de Numpy y se convierte en un grafo de NetworkX. Este
        # caso no genera problemas dado que la conversión a grafo se realiza una sola vez.
        self.graph = nx.from_numpy_array(matrix_data)
        self.num_vertex = self.graph.number_of_nodes()
        # Toolbox permite llamar a los operadores definidos para cada individuo o multiconjunto de individuos.
        self.toolbox = base.Toolbox()
    def evaluate_fill_in(self, individual):
        """
        Evalúa el relleno para un grafo dado un ordenamiento para el mismo. La función calcula la cardinalidad de la
        triangulación resultante del orden de eliminación definido. Se utiliza una copia del grafo original para prevenir
        modificaciones en este.

        El contador de "fill-in" representa el número total de vértices que fueron añadidos para lograr la extensión cordal
        mediante el proceso de eliminación.

        :param individual: Lista ordenada de vértices que representa el orden de eliminación.
        :type individual: list
        :return: Una tupla que contiene el número de aristas agregadas.
        :rtype: tuple
        """
        # Copia del grafo para no alterar el original
        graph_copy = self.graph.copy()
        # Inicializamos el conteo de aristas agregadas en cero
        fill_in_count = 0

        # Iteramos sobre los vértices del grafo
        for vertex in individual:
            # Comparamos por pares los vecinos de cada vertex según el ordenamiento dado
            neighbors = list(graph_copy.neighbors(vertex))
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    u = neighbors[i]
                    v = neighbors[j]

                    # Si u no es adyacente a v, añadimos una arista que los conecte
                    if not graph_copy.has_edge(u, v):
                        graph_copy.add_edge(u, v)
                        # El contador de aristas añadidas aumenta en 1
                        fill_in_count += 1

            # Eliminamos el vértice
            graph_copy.remove_node(vertex)

        # Regresamos el conteo de aristas agregadas (tuple por requisito de DEAP)
        return fill_in_count,
    @ staticmethod
    def SwapMutation(individual):
        """
        SwapMutation (Mutación por Intercambio) es un operador de mutación de algoritmos genéticos. Selecciona
        aleatoriamente dos posiciones distintas en la secuencia de un individuo e intercambia sus valores.

        :param individual: Una secuencia que representa una solución candidata en la población.
        :type individual: list
        :return: Una tupla que contiene el individuo mutado, donde dos elementos de la secuencia han sido intercambiados.
        :rtype: tuple
        """
        size = len(individual)
        a, b = random.sample(range(size), 2)
        individual[a], individual[b] = individual[b], individual[a]

        # DEAP requiere que el resultado sea un objeto tipo tuple.
        return individual,


    # Para utilizar Toolbox adecuadamente, es necesario configurar los operadores requeridos.
    def _setup_toolbox(self):
        """
        Inicializa y configura la "caja de herramientas" (toolbox) del algoritmo genético con las operaciones requeridas,
        tales como la inicialización de la población, los métodos de cruce, las estrategias de mutación y los mecanismos
        de selección.

        La función prepara lo siguiente:
        1. Registra un generador de permutaciones para los índices de los vértices.
        2. Define cómo se estructuran y se crean los individuos y la población inicial.
        3. Establece las operaciones de cruce (PMX), mutación (intercambio simple) y selección (selección por torneo).
        4. Asocia una función de evaluación para determinar la aptitud (fitness) de los individuos.
        """
        # Especificamos la forma en que se generarán las permutaciones de acuerdo al tamaño del grafo.
        self.toolbox.register(alias="indexes", random.sample, range(self.num_vertex), self.num_vertex)

        # Creamos una población inicial como una lista de la estructura antes definida "Individual".
        self.toolbox.register(alias="individual", tools.initIterate, creator.Individual)
        self.toolbox.register(alias="population", tools.initRepeat, list, self.toolbox.individual)

        # Definimos los operadores de cruce (PMX uniforme), mutación (intercambio simple) y selección de descendencia
        # (torneo).
        self.toolbox.register(alias="mate", tools.cxUniformPartialyMatched)
        self.toolbox.register(alias="mutate", self.SwapMutation)
        self.toolbox.register(alias="select_offspring", tools.selTournament, tournsize=5)

        # Evaluamos el fitness de los individuos
        self.toolbox.register(alias="evaluate", self.evaluate_fill_in)