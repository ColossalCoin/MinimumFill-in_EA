import random
import time
import multiprocessing
import networkx as nx
import numpy as np
from deap import creator, base, tools

if not hasattr(creator, "FitnessMin"):  # Evita conflictos por redefinir la misma función de aptitud
    # DEAP requiere que se indique si la función de aptitud debe ser minimizada o maximizada. Declaramos que la función
    # de aptitud debe ser minimizada usando weights=(-1.0,)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    # Creamos un individuo de tipo permutation a partir de una lista. Usar arreglos de numpy no es conveniente en este caso
    # dado que DEAP utiliza Python puro, es decir, requeriría convertir los arreglos en listas cada iteración.
    creator.create("Individual", list, fitness=creator.FitnessMin)


# Definimos el AE como una clase.
class GraphChordalizer:
    def __init__(self, graph):
        """
        Constructor para inicializar una instancia con atributos de datos específicos.

        Inicializa la representación del grafo a partir de la matriz proporcionada usando NetworkX, calcula el número
        de nodos en el grafo y configura una instancia de caja de herramientas (toolbox) para operaciones posteriores.

        :param matrix_data: Matriz de adyacencia cuadrada que define la estructura del grafo. Cada entrada en la matriz
            representa si los vértices correspondientes osn adyacentes.
        :type matrix_data: numpy.ndarray
        """
        # Suponemos que la matriz del constructor es un grafo de NetworkX.
        self.graph = graph
        self.num_vertex = self.graph.number_of_nodes()
        # Toolbox permite llamar a los operadores definidos para cada individuo o multiconjunto de individuos.
        self.toolbox = base.Toolbox()
        self._setup_toolbox()   # Es necesario cargar los operadores desde un inicio
    @staticmethod
    def evaluate_fill_in(individual, graph):
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
        graph_copy = graph.copy()
        # Inicializamos el conteo de aristas agregadas en cero
        fill_in_count = 0

        # Iteramos sobre los vértices del grafo
        for vertex in graph:
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
    def swap_mutation(individual):
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
    @staticmethod
    def roulette_selection(individuals, k):
        """
        Realiza la operación de selección por ruleta para elegir 'k' individuos de la población proporcionada, basándose
        en sus valores de aptitud (fitness). Los individuos con mejor aptitud tienen mayores probabilidades de ser
        seleccionados, pero los individuos con menor aptitud también tienen una probabilidad de selección distinta de cero.

        La aptitud de los individuos se transforma para garantizar que incluso el individuo con menor aptitud pueda
        participar en el proceso de selección. Esto se logra ajustando los valores de aptitud en relación con la aptitud
        del peor individuo en la población.

        :param individuals: Una lista de individuos de la cual se realizará la selección. Cada individuo en la lista debe
            tener un atributo 'fitness' que contenga el valor de aptitud.
        :param k: El número de individuos a seleccionar.
        :return: Una lista de 'k' individuos seleccionados con base en la lógica de selección por ruleta.
        :rtype: list
        """
        # Calculamos la aptitud de cada individuo
        fitnesses = [ind.fitness.values[0] for ind in individuals]

        # Buscamos el candidato menos apto (mayor número de aristas agregadas)
        worst_fitness = max(fitnesses)

        # Invertimos la aptitud con base en el peor individuo. Sumamos 1.0 para que todos los individuos tengan
        # probabilidad no cero de ser seleccionados
        weights = [(worst_fitness - fitness) + 1.0 for fitness in fitnesses]

        # Elegimos k individuos de acuerdo a los pesos asignados.
        chosen = random.choices(individuals, weights=weights, k=k)
        return chosen

    # Para utilizar Toolbox adecuadamente, es necesario configurar los operadores requeridos.
    def _setup_toolbox(self):
        """
        Inicializa y configura la "caja de herramientas" (toolbox) del algoritmo genético con las operaciones requeridas,
        tales como la inicialización de la población, los métodos de cruce, las estrategias de mutación y los mecanismos
        de selección.

        La función prepara lo siguiente:
        1. Registra un generador de permutaciones para los índices de los vértices.
        2. Define cómo se estructuran y se crean los individuos y la población inicial.
        3. Establece las operaciones de selección de progenitores (ruleta), cruce (PMX), mutación (intercambio simple) y
            selección de supervivientes (selección por torneo).
        4. Asocia una función de evaluación para determinar la aptitud (fitness) de los individuos.
        """
        # Especificamos la forma en que se generarán las permutaciones de acuerdo al tamaño del grafo.
        self.toolbox.register("indexes", random.sample, range(self.num_vertex), self.num_vertex)

        # Creamos una población inicial como una lista de la estructura antes definida "Individual".
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.indexes)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Definimos los operadores de selección de progenitores (ruleta) cruce (PMX uniforme), mutación (intercambio
        # simple) y selección de supervivientes (torneo).
        self.toolbox.register("select_parents", self.roulette_selection)
        self.toolbox.register("mate", tools.cxUniformPartialyMatched, indpb=0.5)
        self.toolbox.register("mutate", self.swap_mutation)
        self.toolbox.register("select_offspring", tools.selTournament, tournsize=5)

        # Evaluamos el fitness de los individuos
        self.toolbox.register("evaluate", self.evaluate_fill_in, graph=self.graph)
    def run_ea(self, num_generations=50, population_size=100, crossover_probability=0.7, mutation_probability=0.2,
               verbose=True):
        """
        Ejecuta un algoritmo evolutivo (EA) con los parámetros especificados. La función incluye la generación de la
        población inicial, la evaluación de la aptitud (fitness), las operaciones genéticas (selección, cruce y mutación),
        y rastrea el progreso a través de las generaciones con registro estadístico. También actualiza el Salón de la
        Fama (Hall of Fame, HoF) y registra el tiempo de cálculo para cada generación.

        :param num_generations: El número de generaciones que se ejecutará el EA.
        :type num_generations: int
        :param population_size: El número de individuos en la población.
        :type population_size: int
        :param crossover_probability: La probabilidad con la que se aplica la operación de cruce.
        :type crossover_probability: float
        :param mutation_probability: La probabilidad con la que se aplica la operación de mutación.
        :type mutation_probability: float
        :return: Un libro de registro (logbook) que contiene los registros estadísticos del rendimiento de cada generación.
        :rtype: deap.tools.Logbook
        """
        # Utilizamos paralelización para hacer el código más eficiente
        cpus = multiprocessing.cpu_count()
        # Utilizamos gestión automática de memoria son 'with'
        with multiprocessing.Pool(processes=cpus) as pool:
            # Reemplazamos el 'map' estándar de Python por el 'map' del pool
            self.toolbox.register("map", pool.map)

            # Creamos una población inicial
            population = self.toolbox.population(n=population_size)

            # Evaluamos la población inicial
            fitnesses = list(map(self.toolbox.evaluate, population))
            for individual, fitness in zip(population, fitnesses):
                individual.fitness.values = fitness

            # Registramos estadísticos para monitorear cada población
            stats = tools.Statistics(lambda ind: ind.fitness.values)

            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)   # Peor triangulación
            stats.register("max", np.max)   # Mejor triangulación

            logbook = tools.Logbook()
            logbook.header = ["gen", "avg", "std", "min", "max", "time"]

            if verbose:
                print(f"{'Gen':<5} | {'Min':<5} | {'Avg':<8} | {'Max':<5} | {'Std':<5}", flush=True)

            hof = tools.HallOfFame(1)   # Mejor individuo de todas las generaciones (Hall of fame)

            # Definimos el bucle a ejecutar en cada generación
            for num_generation in range(num_generations):
                # Inicializamos un temporizador para medir el tiempo de ejecución de cada generación
                start_time = time.time()

                # Seleccionamos padres
                offspring = self.toolbox.select_parents(population, len(population))

                # Clonamos los padres seleccionados (necesario en DEAP)
                offspring = list(map(self.toolbox.clone, offspring))

                # Cruzamos los padres
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < crossover_probability:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                # Mutamos a los descendientes
                for mutant in offspring:
                    if random.random() < mutation_probability:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values

                # Evaluamos la descendencia
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(self.toolbox.evaluate, invalid_ind)
                for ind, fitness in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fitness

                # Seleccionamos los supervivientes
                population = self.toolbox.select_offspring(population + offspring, k=population_size)

                # Actualizamos el HoF
                hof.update(population)

                # Compilamos estadísticos
                record = stats.compile(population)

                # Agregamos el tiempo de ejecución de esta generación
                record["time"] = time.time() - start_time

                # Registramos los estadísticos de esta generación
                logbook.record(gen=num_generation, **record)

                if verbose:
                    print(f"{num_generation:<5} | {record['min']:<5.0f} | {record['avg']:<8.2f} | {record['max']:<5.0f} | "
                          f"{record['std']:<5.2f}", flush=True)

        return hof[0], logbook

# --- Probamos que el código funcione correctamente ---
if __name__ == "__main__":
    # Creamos el grafo C_6, cuya triangulación mínima agrega 3 aristas.
    G = nx.cycle_graph(6)

    # Creamos una instancia de la clase GraphChordalizer
    graph_chordalizer = GraphChordalizer(G)

    # Corremos el algoritmo.
    hof, logs = graph_chordalizer.run_ea(num_generations=20, population_size=10)

    print(hof, hof.fitness.values)
