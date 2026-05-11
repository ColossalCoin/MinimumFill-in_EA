import argparse
import sys
import os
import random
import datetime
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.graph_loader import load_pace_graph, nx_to_adj_matrix
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

    # --- BITÁCORA DE INICIO ---
    # Sacamos solo el nombre del archivo (ej. "89.graph") para que el log no sea larguísimo
    nombre_grafo = os.path.basename(args.inst)

    with open("irace_tracking.log", "a") as log_file:
        hora_inicio = datetime.datetime.now().strftime("%H:%M:%S")
        mensaje_inicio = f"[{hora_inicio}] INICIANDO | Grafo: {nombre_grafo} | Pop: {args.pop_size} | Mut: {args.mut_prob:.2f} | Cx: {args.cx_prob:.2f}\n"
        log_file.write(mensaje_inicio)
    # ---------------------------------

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

    cost = best_ind.fitness.values[0]

    # --- BITÁCORA DE FIN ---
    with open("irace_tracking.log", "a") as log_file:
        hora_fin = datetime.datetime.now().strftime("%H:%M:%S")
        mensaje_fin = f"[{hora_fin}] TERMINADO | Grafo: {nombre_grafo} | Costo hallado: {cost}\n"
        log_file.write(mensaje_fin)
    # ---------------------------------

    # 5. Imprimir ÚNICAMENTE el costo. Esto es lo que lee irace.
    print(cost)


if __name__ == "__main__":
    main()