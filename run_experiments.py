import os
import pickle
import pandas as pd
import numpy as np
import time
import random
import multiprocessing
from tqdm import tqdm

# --- IMPORTACIÓN MODULAR ---
from evolution import GraphChordalizer

# ==========================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ==========================================
DATASET_DIR = "datasets"
RESULTS_FILE = "experiment_results.csv"

# Parámetros (Tesis)
N_REPLICAS = 30
POP_SIZE = 50
GENERATIONS = 100
CX_PROB = 0.7
MUT_PROB = 0.2


def run_all_experiments():
    # 1. Obtener lista de grafos
    files = []
    for root, dirs, filenames in os.walk(DATASET_DIR):
        for filename in filenames:
            if filename.endswith(".pkl"):
                files.append(os.path.join(root, filename))
    files.sort()

    print(f"--- Iniciando Experimentos Modulares ---")
    print(f"Grafos: {len(files)} | Réplicas: {N_REPLICAS} | Total: {len(files) * N_REPLICAS}")

    # 2. Preparar archivo de resultados (Resume capability)
    if os.path.exists(RESULTS_FILE):
        existing_df = pd.read_csv(RESULTS_FILE)
        print(f"Resumiendo experimentos. Registros previos: {len(existing_df)}")
    else:
        existing_df = pd.DataFrame()

    results_buffer = []
    total_ops = len(files) * N_REPLICAS

    pbar = tqdm(total=total_ops, desc="Progreso Global")

    for filepath in files:
        category = os.path.basename(os.path.dirname(filepath))
        name = os.path.basename(filepath).replace(".pkl", "")

        # Cargar grafo
        with open(filepath, 'rb') as f:
            G = pickle.load(f)

        for replica in range(N_REPLICAS):
            # Verificar si ya existe
            if not existing_df.empty:
                done = existing_df[
                    (existing_df["Graph"] == name) &
                    (existing_df["Replica"] == replica)
                    ]
                if not done.empty:
                    pbar.update(1)
                    continue

            # --- SEMILLA DETERMINISTA ---
            seed_val = (hash(name) + replica) % (2 ** 32)
            random.seed(seed_val)
            np.random.seed(seed_val)

            # --- EJECUCIÓN (Usando la clase importada) ---
            start_total = time.time()

            # Instanciar
            optimizer = GraphChordalizer(G)

            # Correr (Verbose False para mantener limpia la consola)
            best_ind, logbook = optimizer.run_ea(
                num_generations=GENERATIONS,
                population_size=POP_SIZE,
                crossover_probability=CX_PROB,
                mutation_probability=MUT_PROB,
                verbose=False
            )

            total_time = time.time() - start_total

            # Extraer métricas
            best_fillin = best_ind.fitness.values[0]
            min_history = logbook.select("min")
            # Encontrar primera generación donde se logró el óptimo
            gen_conv = next((i for i, v in enumerate(min_history) if v == best_fillin), GENERATIONS)

            results_buffer.append({
                "Category": category,
                "Graph": name,
                "Nodes": G.number_of_nodes(),
                "Edges": G.number_of_edges(),
                "Replica": replica,
                "Best_FillIn": best_fillin,
                "Gen_Found": gen_conv,
                "Time_Total_s": round(total_time, 4),
                "Pop_Size": POP_SIZE,
                "Generations": GENERATIONS
            })

            pbar.update(1)

        # Guardado incremental por Grafo
        if results_buffer:
            new_df = pd.DataFrame(results_buffer)
            mode = 'a' if os.path.exists(RESULTS_FILE) else 'w'
            header = not os.path.exists(RESULTS_FILE)
            new_df.to_csv(RESULTS_FILE, mode=mode, header=header, index=False)
            results_buffer = []

    pbar.close()
    print(f"\n✅ Resultados guardados en {RESULTS_FILE}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_all_experiments()