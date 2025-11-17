import random
import networkx as nx
from deap import creator, base, tools

def evaluate_fill_in(individual, graph):
    # Copia del grafo para no alterar el original
    graph_copy = graph.copy()
    # Inicializamos el conteo de aristas agregadas en cero
    fill_in_count = 0

    # Iteramos sobre los vértices del grafo
    for vertex in individual:
        # Comparamos por pares los vecinos de cada vertex según el ordenamiento dado
        neighbors = list(graph_copy.neighbors(vertex))
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
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

# Declaramos el objetivo de la optimizacion (minimizar)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# Creamos un individuo de tipo permutation con longitud IND_SIZE=10
creator.create("Individual", np.ndarray, dtype="d", fitness=creator.FitnessMin)

IND_SIZE = 10

toolbox = base.Toolbox()
toolbox.register("attr_float", random.randint, 0, 1)
toolbox.register("Individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)

# Creamos una population
toolbox.register("population", tools.initRepeat, list, toolbox.Individual)

# Elegimos el operador de cruce
toolbox.register("mate", tools.cxUniformPartialyMatched)

# Elegimos el operador de mutation
toolbox.register("mutate", tools.p)

# Elegimos el operador de selection
toolbox.register("select", tools.selTournament, k=5, tournsize=5)

# Definimos el AE