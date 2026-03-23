import argparse
import sys
import random
import numpy as np
from pathlib import Path

# Añadimos el root al path, igual que en tus scripts
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_loader import load_pace_graph, nx_to_adj_matrix
from src.graph_chordalizer import GraphChordalizer


def main():
    parser = argparse.ArgumentParser(description="Wrapper de irace para AE de Fill-in")
    # Argumentos que irace enviará por defecto o configurados
    parser.add_argument("--inst", type=str, required=True, help="Ruta de la instancia (grafo)")
    parser.add_argument("--seed", type=int, help="Semilla de irace para reproducibilidad")

    # Tus hiperparámetros a calibrar
    parser.add_argument("--pop_size", type=int, required=True)
    parser.add_argument("--cx_prob", type=float, required=True)
    parser.add_argument("--mut_prob", type=float, required=True)
    parser.add_argument("--tournsize", type=int, required=True)

    # Usamos parse_known_args por si irace envía argumentos ocultos (como --bound)
    args, _ = parser.parse_known_args()

    # 1. Fijar semillas para garantizar la reproducibilidad estadística en irace
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    # 2. Cargar el grafo y convertirlo a matriz de adyacencia
    G = load_pace_graph(args.inst)
    if G is None:
        # Si falla la carga, imprimimos "inf" para que irace descarte esta ejecución
        print("inf")
        sys.exit(1)

    A = nx_to_adj_matrix(G)

    # 3. Inicializar el algoritmo
    ea = GraphChordalizer(A)

    # 4. Ejecutar el AE (Usamos un presupuesto fijo para que la comparación sea justa)
    best_ind, logbook = ea.run_ea(
        num_generations=100,  # Límite superior holgado
        population_size=args.pop_size,
        cx_prob=args.cx_prob,
        mut_prob=args.mut_prob,
        tournsize=args.tournsize,
        max_evaluations=2000,  # IMPORTANTE: Presupuesto fijo de evaluaciones
        verbose=False
    )

    # 5. Imprimir ÚNICAMENTE el costo. Esto es lo que lee irace.
    cost = best_ind.fitness.values[0]
    print(cost)


if __name__ == "__main__":
    main()