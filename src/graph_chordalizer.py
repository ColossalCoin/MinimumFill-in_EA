import random
import time
import sys
import multiprocessing
from pathlib import Path

import numpy as np
from deap import creator, base, tools

# Aseguramos que el directorio raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.heuristics import count_fillin

if not hasattr(creator, "FitnessMin"):  # Evita conflictos por redefinir la misma función de aptitud
    # DEAP requiere que se indique si la función de aptitud debe ser minimizada o maximizada. Declaramos que la función
    # de aptitud debe ser minimizada usando weights=(-1.0,)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    # Creamos un individuo de tipo permutation a partir de una lista. Usar arreglos de numpy no es conveniente en este caso
    # dado que DEAP utiliza Python puro, es decir, requeriría convertir los arreglos en listas cada iteración.
    creator.create("Individual", list, fitness=creator.FitnessMin)


# Definimos el AE como una clase.
class GraphChordalizer:
    def __init__(self, adj_matrix: np.ndarray, tournsize: int = 3):
        """
        Constructor para inicializar una instancia con atributos de datos específicos.

        Inicializa la representación del grafo a partir de una matriz de adyacencia NumPy, calcula el número
        de nodos en el grafo y configura una instancia de caja de herramientas (toolbox) para operaciones posteriores.

        :param adj_matrix: Matriz de adyacencia cuadrada (n x n). Cada entrada indica si los vértices correspondientes
            son adyacentes. Se asume grafo no dirigido (simétrico) y sin autolazos (diagonal en 0).
        :type adj_matrix: numpy.ndarray
        :param tournsize: Tamaño del torneo para la selección de supervivientes.
        :type tournsize: int
        """
        A = np.asarray(adj_matrix)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("adj_matrix debe ser una matriz cuadrada (n x n).")

        # Guardamos una copia booleana ligera; no se modifica durante el AE (solo se copia en evaluación).
        self.adj_matrix = (A != 0)
        # Normalizamos una sola vez para evitar validaciones/costos por evaluación.
        if not np.array_equal(self.adj_matrix, self.adj_matrix.T):
            self.adj_matrix |= self.adj_matrix.T
        self.adj_matrix[np.arange(self.adj_matrix.shape[0]), np.arange(self.adj_matrix.shape[0])] = False
        self.num_vertex = int(self.adj_matrix.shape[0])
        self.tournsize = int(tournsize)
        # Toolbox permite llamar a los operadores definidos para cada individuo o multiconjunto de individuos.
        self.toolbox = base.Toolbox()
        self._setup_toolbox()   # Es necesario cargar los operadores desde un inicio

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
        self.toolbox.register("select_offspring", tools.selTournament, tournsize=self.tournsize)

        # Evaluamos el fitness de los individuos
        self.toolbox.register("evaluate", self.eval_wrapper, adj_matrix=self.adj_matrix)

    @staticmethod
    def eval_wrapper(individual, adj_matrix):
        """Wrapper para adaptar la firma de count_fillin a lo que pide DEAP (tupla)"""
        # count_fillin devuelve un int, DEAP necesita (int, )
        return count_fillin(adj_matrix, individual, validate=False),

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
        return random.choices(individuals, weights=weights, k=k)

    # Para utilizar Toolbox adecuadamente, es necesario configurar los operadores requeridos.

    def run_ea(
        self,
        num_generations=50,
        population_size=100,
        cx_prob=0.7,
        mut_prob=0.2,
        max_evaluations=None,
        verbose=False,
    ):
        """
        Ejecuta el algoritmo evolutivo (EA) con opción de limitar el número total de evaluaciones de fitness.

        :param num_generations: Número máximo de generaciones a ejecutar.
        :param population_size: Tamaño de la población.
        :param cx_prob: Probabilidad de cruce.
        :param mut_prob: Probabilidad de mutación.
        :param max_evaluations: Presupuesto máximo de evaluaciones de fitness. Si es None,
            se ignora y el control se hace solo por número de generaciones.
        :param verbose: Si es True, imprime información de progreso.
        """

        # Configuración de estadísticas
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)

        logbook = tools.Logbook()

        # POOL: Manejo seguro para Windows/Linux
        pool = None
        try:
            # Solo usar multiproceso si el grafo es grande (>50 nodos), si no el overhead gana
            if self.num_vertex > 50:
                pool = multiprocessing.Pool()
                self.toolbox.register("map", pool.map)

            # --- ALGORITMO EVOLUTIVO (Mu + Lambda) ---

            # 1. Población Inicial
            pop = self.toolbox.population(n=population_size)

            # Evaluación inicial
            fitnesses = list(map(self.toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            evaluations = len(pop)

            hof = tools.HallOfFame(1)
            hof.update(pop)

            for gen in range(num_generations):
                # Si ya alcanzamos el presupuesto, detenemos el ciclo
                if max_evaluations is not None and evaluations >= max_evaluations:
                    break

                # Selección de padres (toda la población compite)
                offspring = self.toolbox.select_parents(pop, len(pop))
                offspring = list(map(self.toolbox.clone, offspring))

                # Cruce
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < cx_prob:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                # Mutación
                for mutant in offspring:
                    if random.random() < mut_prob:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values

                # Evaluación de nuevos candidatos
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                evaluations += len(invalid_ind)

                # Supervivencia: Selección elitista combinando Padres + Hijos
                pop = self.toolbox.select_offspring(pop + offspring, k=population_size)

                # Guardar estadísticas
                hof.update(pop)
                record = stats.compile(pop)
                record["evals"] = evaluations
                logbook.record(gen=gen, **record)

                if verbose:
                    print(f"Gen {gen}: Mejor {record['min']:.0f} | Evals {evaluations}")

        finally:
            if pool:
                pool.close()
                pool.join()
                self.toolbox.unregister("map")

        return hof[0], logbook


# --- Probamos que el código funcione correctamente ---
if __name__ == "__main__":
    # Grafo ciclo C_6 en matriz de adyacencia, cuya triangulación mínima agrega 3 aristas.
    n = 6
    A = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        A[i, (i + 1) % n] = 1
        A[(i + 1) % n, i] = 1

    # Creamos una instancia de la clase GraphChordalizer
    graph_chordalizer = GraphChordalizer(A)

    # Corremos el algoritmo.
    hof, logs = graph_chordalizer.run_ea(num_generations=20, population_size=10)

    print(hof, hof.fitness.values)
