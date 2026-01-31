import pandas as pd
import itertools
from tqdm.auto import tqdm
import sys

sys.path.append('..')
from src.graph_chordalizer import GraphChordalizer


class GridSearchTuner:
    def __init__(self, graph, param_grid, repetitions=5, generations=50):
        """
        Clase orquestadora para la calibración de hiperparámetros.

        :param graph: Grafo NetworkX (Instancia de Sacrificio).
        :param param_grid: Diccionario con listas de valores.
                           Ej: {'pop_size': [50, 100], 'mut_prob': [0.1]}
        :param repetitions: Veces que se repite cada configuración (para robustez estadística).
        :param generations: Número de generaciones fijas para la prueba.
        """
        self.graph = graph
        self.param_grid = param_grid
        self.repetitions = repetitions
        self.generations = generations
        self.results = []

    def run(self, verbose=True):
        """
        Ejecuta el Grid Search completo.
        """
        # 1. Generar todas las combinaciones posibles (Producto Cartesiano)
        keys, values = zip(*self.param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        if verbose:
            print(f"Iniciando Grid Search con {len(combinations)} configuraciones.")
            print(f"Repeticiones por config: {self.repetitions}")
            print(f"Total de ejecuciones: {len(combinations) * self.repetitions}")

        # 2. Iterar sobre configuraciones (con barra de progreso)
        # Usamos tqdm para ver cuánto falta
        iterator = tqdm(combinations, desc="Calibrando") if verbose else combinations

        for config in iterator:
            metrics_run = []

            # 3. Repeticiones para robustez
            for i in range(self.repetitions):
                # Instanciamos el algoritmo
                ga = GraphChordalizer(self.graph)

                # Ejecutamos (muteado para no ensuciar consola)
                best_ind, _ = ga.run_ea(
                    num_generations=self.generations,
                    population_size=config['pop_size'],
                    cx_prob=config.get('cx_prob', 0.7),  # Default seguro
                    mut_prob=config.get('mut_prob', 0.2),
                    verbose=False
                )

                # Guardamos el fitness final
                metrics_run.append(best_ind.fitness.values[0])

            # 4. Calcular estadísticas de la configuración
            avg_fitness = sum(metrics_run) / len(metrics_run)
            best_fitness = min(metrics_run)
            worst_fitness = max(metrics_run)

            # 5. Guardar registro
            result_row = config.copy()
            result_row.update({
                'Avg_Best_Fitness': avg_fitness,
                'Min_Fitness': best_fitness,
                'Max_Fitness': worst_fitness,
                'Repetitions': self.repetitions
            })
            self.results.append(result_row)

        if verbose:
            print("Calibración finalizada.")

        return self.get_results_dataframe()

    def get_results_dataframe(self):
        """Retorna los resultados como DataFrame ordenado por mejor desempeño."""
        df = pd.DataFrame(self.results)
        if not df.empty:
            df = df.sort_values(by='Avg_Best_Fitness', ascending=True).reset_index(drop=True)
        return df