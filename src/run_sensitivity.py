import os
import pickle
import pandas as pd
import numpy as np
import time
import random
import multiprocessing
from chordal_ga import GraphChordalizer

# Seleccionamos solo unos pocos grafos representativos para este análisis profundo
SELECTED_GRAPHS = ["grid8x8_N64", "bcsstk02", "grafo_denso_ejemplo"]

DATASET_DIR = "datasets"
RESULTS_FILE = "sensitivity_results.csv"

# Definimos los escenarios a probar (Grid Search)
SCENARIOS = [
    {"pop": 20, "mut": 0.1, "cx": 0.6},
    {"pop": 20, "mut": 0.5, "cx": 0.9},  # Escenario agresivo
    {"pop": 50, "mut": 0.2, "cx": 0.7},  # Estándar (Control)
    {"pop": 100, "mut": 0.1, "cx": 0.8},  # Población grande, conservador
    {"pop": 100, "mut": 0.4, "cx": 0.6},
]

N_REPLICAS = 30  # Conservando significancia estadística
GENERATIONS = 100  # Fijo para comparar MBF (Mean Best Fitness)


def get_target_fitness(graph_name):
    """
    Define la meta de calidad.
    Lee el valor de 'baselines.csv' (el valor de MinDegree).
    """
    baseline_df = pd.read_csv("baselines.csv")
    return baseline_df.loc[graph_name, "Base_MinDegree"]


def run_sensitivity_analysis():
    results = []

    # Buscar archivos
    files = []
    for root, dirs, filenames in os.walk(DATASET_DIR):
        for filename in filenames:
            name = filename.replace(".pkl", "")
            # Solo procesamos los grafos seleccionados
            if name in SELECTED_GRAPHS:
                files.append(os.path.join(root, filename))

    print(f"--- Iniciando Análisis de Sensibilidad (Eiben) ---")
    print(f"Grafos: {len(files)} | Escenarios: {len(SCENARIOS)} | Réplicas: {N_REPLICAS}")

    for filepath in files:
        name = os.path.basename(filepath).replace(".pkl", "")

        # Cargar grafo
        with open(filepath, 'rb') as f:
            G = pickle.load(f)

        # Target para medir Success Rate (Opcional si tienes los benchmarks cargados)
        target_val = get_target_fitness(name)

        for scenario_idx, params in enumerate(SCENARIOS):
            print(f" > Procesando {name} | Escenario {scenario_idx}: {params}")

            for replica in range(N_REPLICAS):
                # Semilla
                seed_val = (hash(name) + replica + scenario_idx) % (2 ** 32)
                random.seed(seed_val)
                np.random.seed(seed_val)

                # Ejecución
                optimizer = GraphChordalizer(G)
                start_time = time.time()

                best_ind, logbook = optimizer.run_ea(
                    num_generations=GENERATIONS,
                    population_size=params["pop"],
                    crossover_probability=params["cx"],
                    mutation_probability=params["mut"],
                    verbose=False
                )

                elapsed = time.time() - start_time
                best_fillin = best_ind.fitness.values[0]

                # --- MÉTRICAS EIBEN ---
                # 1. AES (Average Evaluations to Solution)
                # Buscamos en qué evaluación se logró el éxito (si se logró)
                # Estimado: gen_found * pop_size
                min_history = logbook.select("min")
                gen_found = next((i for i, v in enumerate(min_history) if v <= target_val), GENERATIONS)

                success = 1 if best_fillin <= target_val else 0
                evals_to_solution = gen_found * params["pop"] if success else GENERATIONS * params["pop"]

                results.append({
                    "Graph": name,
                    "Replica": replica,
                    "Pop_Size": params["pop"],
                    "Mut_Rate": params["mut"],
                    "CX_Rate": params["cx"],
                    "Best_FillIn": best_fillin,
                    "Success": success,
                    "AES": evals_to_solution,  # Average Evaluations to Solution
                    "Time_s": elapsed
                })

    # Guardar
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"Análisis de sensibilidad guardado en {RESULTS_FILE}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_sensitivity_analysis()