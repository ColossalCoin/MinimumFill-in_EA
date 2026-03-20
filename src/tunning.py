import pandas as pd
import numpy as np
import itertools
from tqdm.auto import tqdm
import sys
from pathlib import Path

# Aseguramos que el directorio raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_chordalizer import GraphChordalizer


class GridSearchTuner:
    def __init__(self, adj_matrix: np.ndarray, param_grid: dict, repetitions: int = 5, max_generations: int = 50, eval_budget: int | None = None):
        """
        Clase orquestadora para la calibración de hiperparámetros.

        :param adj_matrix: Matriz de adyacencia NumPy (instancia de sacrificio).
        :param param_grid: Diccionario con listas de valores.
                           Ej: {'pop_size': [50, 100], 'mut_prob': [0.1], 'tournsize': [3, 5]}
        :param repetitions: Veces que se repite cada configuración (para robustez estadística).
        :param max_generations: Número máximo de generaciones por ejecución.
        :param eval_budget: Presupuesto fijo de evaluaciones de fitness para todas las configuraciones.
        """
        self.adj_matrix = np.asarray(adj_matrix)
        self.param_grid = param_grid
        self.repetitions = repetitions
        self.max_generations = max_generations
        self.results = []   # Lista plana de diccionarios

        # Definir presupuesto de evaluaciones común
        if eval_budget is not None:
            self.eval_budget = int(eval_budget)
        else:
            if "pop_size" not in self.param_grid:
                raise ValueError("param_grid debe contener la clave 'pop_size' o se debe especificar eval_budget.")
            max_pop = max(self.param_grid["pop_size"])
            self.eval_budget = int(self.max_generations * max_pop)

    def run(self, verbose=True):
        """
        Ejecuta el Grid Search y devuelve datos crudos.
        """
        keys, values = zip(*self.param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        if verbose:
            total_runs = len(combinations) * self.repetitions
            print(
                f"Iniciando Tuning: {len(combinations)} configs x {self.repetitions} reps = {total_runs} ejecuciones.")

        # Barra de progreso total
        with tqdm(total=len(combinations) * self.repetitions, desc="Running Runs") as pbar:

            for config_id, config in enumerate(combinations):

                for rep in range(self.repetitions):
                    # Ejecutar EA
                    ea = GraphChordalizer(self.adj_matrix)
                    best_ind, logbook = ea.run_ea(
                        num_generations=self.max_generations,
                        population_size=config['pop_size'],
                        cx_prob=config.get('cx_prob', 0.8),
                        mut_prob=config.get('mut_prob', 0.2),
                        tournsize=config.get('tournsize', 3),
                        max_evaluations=self.eval_budget,
                        verbose=False
                    )

                    # Guardamos cada dato individualmente
                    # Aplanamos el diccionario de config + resultados del run
                    record = config.copy()
                    record.update({
                        'config_id': config_id,  # ID único para agrupar luego
                        'run_id': rep,  # Número de repetición
                        'fitness': best_ind.fitness.values[0],
                        'generations': len(logbook),
                        'eval_budget': self.eval_budget,
                        'evals_used': logbook[-1]['evals'] if len(logbook) > 0 and 'evals' in logbook[-1] else None,
                    })

                    self.results.append(record)
                    pbar.update(1)

        return pd.DataFrame(self.results)