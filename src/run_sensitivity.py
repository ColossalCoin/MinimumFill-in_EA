#!/usr/bin/env python3
"""
run_sensitivity.py
==================
Ejecuta el análisis de sensibilidad OFAT del AE para el problema de relleno mínimo.

Para cada instancia del benchmark y cada configuración OFAT, lanza N_RUNS ejecuciones
independientes y guarda en disco:
  - Un JSON por par (instancia, configuración) con el historial completo por generación.
  - Un CSV de resumen plano al finalizar, listo para cargar en el notebook de análisis.

Requisitos previos
------------------
- Aplicar los tres cambios indicados en graph_chordalizer.py (use_multiprocessing,
  registro gen=-1 en logbook).
- Los archivos de grafos deben estar en DATASETS_DIR con el patrón de nombres de PACE 2017
  (p. ej. exact_009.gr, exact_051.gr, …). Ajusta FILE_TEMPLATE si tu convención difiere.

Uso
---
    python run_sensitivity.py                      # usa todas las instancias y 30 runs
    python run_sensitivity.py --runs 5             # prueba rápida con 5 runs
    python run_sensitivity.py --graphs 009 040     # solo esas instancias
    python run_sensitivity.py --no_resume          # reescribe resultados existentes
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── Aseguramos que el directorio raíz del proyecto esté en sys.path ──────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph_chordalizer import GraphChordalizer
from utils.graph_loader import load_pace_graph, nx_to_adj_matrix
from utils.heuristics import greedy_minimum_degree

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════════════════════

DATASETS_DIR = Path(__file__).resolve().parent.parent / "data" / "train"
RESULTS_DIR  = Path(__file__) / "processed"

N_RUNS          = 30        # Ejecuciones independientes por (instancia, config)
MAX_EVALUATIONS = 2_000     # Presupuesto fijo, igual que en sintonización
NUM_GENERATIONS = 10_000    # Techo alto; el control real lo lleva MAX_EVALUATIONS

# Patrón de nombre de archivo PACE 2017.  {id} se reemplaza por el ID de 3 dígitos.
# Cambia este template si tu convención es distinta (p. ej. "minfill_{id}.gr").
FILE_TEMPLATE = "{id}.graph"

# ── Instancias del benchmark (Cuadro 3.1) ─────────────────────────────────────
BENCHMARK_INSTANCES: list[dict] = [
    {"id": "009", "category": "small"},
    {"id": "051", "category": "small"},
    {"id": "003", "category": "medium"},
    {"id": "026", "category": "medium"},
    {"id": "040", "category": "medium"},
    {"id": "089", "category": "medium"},
    {"id": "092", "category": "medium"},
    {"id": "056", "category": "large"},
    {"id": "065", "category": "large"},
    {"id": "071", "category": "large"},
    {"id": "072", "category": "large"},
    {"id": "094", "category": "large"},
]

# ── Configuración óptima encontrada por irace (Cuadro 4.1) ────────────────────
OPTIMAL_CONFIG: dict = {
    "population_size": 50,
    "cx_prob":         0.7544,
    "mut_prob":        0.8577,
    "tournsize":       5,
}

# ── Niveles OFAT: 4 perturbaciones por parámetro (L1 < L2 < Θ* < L3 < L4) ────
OFAT_LEVELS: dict[str, list] = {
    "population_size": [25, 38, 65, 100],
    "cx_prob":         [0.60, 0.68, 0.83, 0.95],
    "mut_prob":        [0.50, 0.70, 0.93, 0.99],
    "tournsize":       [2, 3, 7, 9],
}

# ═══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════

def build_configs() -> list[dict]:
    """Construye la lista de 17 configuraciones OFAT (1 óptima + 16 perturbadas)."""
    configs = [{"name": "optimal", **OPTIMAL_CONFIG}]
    for param, levels in OFAT_LEVELS.items():
        for level in levels:
            cfg = {**OPTIMAL_CONFIG, param: level}
            # Nombre legible: parámetro_valor (reemplaza '.' por 'p' para nombres de archivo)
            level_str = str(level).replace(".", "p")
            cfg["name"] = f"{param}_{level_str}"
            configs.append(cfg)
    return configs


def find_graph_file(datasets_dir: Path, graph_id: str) -> Optional[Path]:
    """
    Busca el archivo de grafo por ID probando patrones comunes de PACE 2017.
    Devuelve None si no lo encuentra.
    """
    candidates = [
        datasets_dir / FILE_TEMPLATE.format(id=graph_id),
        datasets_dir / f"{graph_id}.gr",
        datasets_dir / f"minfill_{graph_id}.gr",
        datasets_dir / f"exact{graph_id}.gr",
    ]
    for path in candidates:
        if path.exists():
            return path
    # Búsqueda por glob como último recurso
    matches = sorted(datasets_dir.glob(f"*{graph_id}*"))
    return matches[0] if matches else None


def compute_baseline(adj_matrix: np.ndarray) -> tuple[int, list[int]]:
    """
    Calcula la solución de referencia (mínimo grado) para una instancia.
    Devuelve (costo_fillin, orden_eliminacion).
    """
    order, cost = greedy_minimum_degree(adj_matrix, compute_cost=True)
    return int(cost), order


def _run_single_experiment(
    adj_matrix: np.ndarray,
    config: dict,
    baseline: int,
    run_id: int,
) -> dict:
    """
    Ejecuta una sola ejecución del AE y devuelve el registro completo.

    Parámetros
    ----------
    adj_matrix : matriz de adyacencia de la instancia.
    config     : diccionario con population_size, cx_prob, mut_prob, tournsize.
    baseline   : costo de referencia (mínimo grado).
    run_id     : índice de la ejecución (0..N_RUNS-1), usado como semilla base.
    """
    # Semilla reproducible pero diferente por ejecución
    seed = run_id * 1_000_003 % (2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)

    chordalizer = GraphChordalizer(adj_matrix)

    t0 = time.perf_counter()
    hof, logbook = chordalizer.run_ea(
        num_generations    = NUM_GENERATIONS,
        population_size    = config["population_size"],
        cx_prob            = config["cx_prob"],
        mut_prob           = config["mut_prob"],
        tournsize          = config["tournsize"],
        max_evaluations    = MAX_EVALUATIONS,
        verbose            = False,
        use_multiprocessing= False,   # Deshabilitado para evitar overhead en experimentos secuenciales
    )
    runtime = time.perf_counter() - t0

    # ── Extrae columnas del logbook ──────────────────────────────────────────
    gens  = logbook.select("gen")
    evals = logbook.select("evals")
    mins_ = logbook.select("min")
    avgs_ = logbook.select("avg")

    final_best = int(hof.fitness.values[0])
    success    = bool(final_best <= baseline)

    # AES: primer evaluación en que el mejor fitness alcanza la referencia
    aes: Optional[int] = None
    for e, m in zip(evals, mins_):
        if m <= baseline:
            aes = int(e)
            break

    return {
        "run_id"       : run_id,
        "seed"         : seed,
        "gens"         : [int(g) for g in gens],
        "evals"        : [int(e) for e in evals],
        "best_per_gen" : [float(m) for m in mins_],
        "avg_per_gen"  : [float(a) for a in avgs_],
        "final_best"   : final_best,
        "success"      : success,
        "aes"          : aes,
        "runtime_sec"  : round(runtime, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL DE EXPERIMENTO
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    instance: dict,
    config: dict,
    adj_matrix: np.ndarray,
    baseline: int,
    n_runs: int,
    output_path: Path,
    resume: bool = True,
) -> dict:
    """
    Lanza n_runs ejecuciones para un par (instancia, config) y guarda el resultado.

    Si resume=True y output_path ya existe, carga y devuelve el resultado existente.
    """
    if resume and output_path.exists():
        with open(output_path) as f:
            return json.load(f)

    runs = []
    for run_id in range(n_runs):
        run_result = _run_single_experiment(adj_matrix, config, baseline, run_id)
        runs.append(run_result)

        # Progreso en línea
        status = "✓" if run_result["success"] else "✗"
        aes_str = str(run_result["aes"]) if run_result["aes"] else "---"
        print(
            f"    run {run_id+1:02d}/{n_runs}  "
            f"best={run_result['final_best']:>5}  "
            f"baseline={baseline:>5}  "
            f"{status}  AES={aes_str:>6}  "
            f"t={run_result['runtime_sec']:.2f}s",
            flush=True,
        )

    # ── Estadísticas agregadas ───────────────────────────────────────────────
    finals    = [r["final_best"] for r in runs]
    successes = [r["success"]    for r in runs]
    aes_vals  = [r["aes"] for r in runs if r["aes"] is not None]

    summary = {
        "mbf"       : float(np.mean(finals)),
        "mbf_std"   : float(np.std(finals, ddof=1)),
        "sr"        : float(np.mean(successes)),
        "aes_mean"  : float(np.mean(aes_vals))  if aes_vals else None,
        "aes_std"   : float(np.std(aes_vals, ddof=1)) if len(aes_vals) > 1 else None,
        "n_success" : int(sum(successes)),
        "n_runs"    : n_runs,
    }

    result = {
        "graph_id"   : instance["id"],
        "category"   : instance["category"],
        "n_vertices" : int(adj_matrix.shape[0]),
        "baseline"   : int(baseline),
        "config_name": config["name"],
        "config"     : {k: v for k, v in config.items() if k != "name"},
        "summary"    : summary,
        "runs"       : runs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DEL CSV DE RESUMEN
# ═══════════════════════════════════════════════════════════════════════════════

def build_summary_csv(results_dir: Path) -> Path:
    """
    Recorre todos los JSON en results_dir y construye un CSV plano con una fila
    por (instancia, config, run).  Ideal para cargar directamente con pandas.
    """
    import csv

    rows = []
    for json_path in sorted(results_dir.glob("*.json")):
        with open(json_path) as f:
            data = json.load(f)

        graph_id   = data["graph_id"]
        category   = data["category"]
        n_vertices = data["n_vertices"]
        baseline   = data["baseline"]
        cfg_name   = data["config_name"]
        cfg        = data["config"]

        for run in data["runs"]:
            rows.append({
                "graph_id"        : graph_id,
                "category"        : category,
                "n_vertices"      : n_vertices,
                "baseline"        : baseline,
                "config_name"     : cfg_name,
                "population_size" : cfg["population_size"],
                "cx_prob"         : cfg["cx_prob"],
                "mut_prob"        : cfg["mut_prob"],
                "tournsize"       : cfg["tournsize"],
                "run_id"          : run["run_id"],
                "final_best"      : run["final_best"],
                "success"         : int(run["success"]),
                "aes"             : run["aes"] if run["aes"] is not None else "",
                "runtime_sec"     : run["runtime_sec"],
                # Gap relativo respecto a la referencia de mínimo grado
                "gap_pct"         : round(
                    100.0 * (run["final_best"] - baseline) / max(baseline, 1), 2
                ),
            })

    csv_path = results_dir / "summary.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[CSV] Resumen guardado en: {csv_path}  ({len(rows)} filas)")
    else:
        print("\n[CSV] No se encontraron resultados para resumir.")

    return csv_path


# ═══════════════════════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Análisis de sensibilidad OFAT del AE.")
    p.add_argument(
        "--datasets_dir", type=Path, default=DATASETS_DIR,
        help="Directorio con los archivos .gr de PACE 2017."
    )
    p.add_argument(
        "--results_dir", type=Path, default=RESULTS_DIR,
        help="Directorio donde se guardarán los resultados."
    )
    p.add_argument(
        "--runs", type=int, default=N_RUNS,
        help="Número de ejecuciones independientes por configuración."
    )
    p.add_argument(
        "--graphs", nargs="+", default=None,
        help="IDs de grafos a procesar (p. ej. --graphs 009 040). "
             "Si se omite, se procesan todos los del benchmark."
    )
    p.add_argument(
        "--configs", nargs="+", default=None,
        help="Nombres de configuraciones a ejecutar (p. ej. --configs optimal mu_25). "
             "Si se omite, se ejecutan todas las configuraciones OFAT."
    )
    p.add_argument(
        "--no_resume", action="store_true",
        help="Reescribe resultados existentes en lugar de saltarlos."
    )
    p.add_argument(
        "--only_csv", action="store_true",
        help="Solo regenera el CSV de resumen a partir de JSONs existentes."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results_dir  = args.results_dir
    datasets_dir = args.datasets_dir
    resume       = not args.no_resume

    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Solo CSV ────────────────────────────────────────────────────────────
    if args.only_csv:
        build_summary_csv(results_dir)
        return

    # ── Instancias a procesar ────────────────────────────────────────────────
    instances = BENCHMARK_INSTANCES
    if args.graphs:
        requested = set(args.graphs)
        instances = [i for i in instances if i["id"] in requested]
        if not instances:
            print(f"[ERROR] Ningún ID solicitado coincide con el benchmark: {args.graphs}")
            sys.exit(1)

    # ── Configuraciones a ejecutar ───────────────────────────────────────────
    all_configs = build_configs()
    if args.configs:
        requested_cfg = set(args.configs)
        all_configs = [c for c in all_configs if c["name"] in requested_cfg]
        if not all_configs:
            print(f"[ERROR] Ninguna configuración solicitada es válida: {args.configs}")
            sys.exit(1)

    n_runs = args.runs

    # ── Resumen del plan ─────────────────────────────────────────────────────
    total = len(instances) * len(all_configs) * n_runs
    print("=" * 70)
    print(f"  Instancias  : {len(instances)}")
    print(f"  Configs OFAT: {len(all_configs)}")
    print(f"  Runs/config : {n_runs}")
    print(f"  Total runs  : {total}")
    print(f"  Presupuesto : {MAX_EVALUATIONS} evaluaciones/run")
    print(f"  Resultados  : {results_dir}")
    print("=" * 70)

    all_results: list[dict] = []
    t_global = time.perf_counter()

    for inst_idx, instance in enumerate(instances, 1):
        graph_id = instance["id"]
        category = instance["category"]

        # ── Carga del grafo ──────────────────────────────────────────────────
        graph_path = find_graph_file(datasets_dir, graph_id)
        if graph_path is None:
            print(f"\n[SKIP] Grafo {graph_id}: archivo no encontrado en {datasets_dir}")
            continue

        G = load_pace_graph(str(graph_path))
        if G is None:
            print(f"\n[SKIP] Grafo {graph_id}: error al cargar {graph_path}")
            continue

        adj_matrix = nx_to_adj_matrix(G)
        n_vertices = int(adj_matrix.shape[0])

        # ── Línea base (mínimo grado) ────────────────────────────────────────
        baseline, _ = compute_baseline(adj_matrix)

        print(f"\n{'─'*70}")
        print(
            f"  [{inst_idx}/{len(instances)}] Grafo {graph_id} | "
            f"{category} | {n_vertices} vértices | baseline MD = {baseline}"
        )
        print(f"{'─'*70}")

        for cfg_idx, config in enumerate(all_configs, 1):
            output_path = results_dir / f"graph_{graph_id}_{config['name']}.json"

            print(
                f"\n  Config [{cfg_idx}/{len(all_configs)}]: {config['name']}"
                f"  (μ={config['population_size']}, "
                f"pc={config['cx_prob']:.2f}, "
                f"pm={config['mut_prob']:.2f}, "
                f"k={config['tournsize']})"
            )

            if resume and output_path.exists():
                print("    → Resultado existente, omitiendo (usa --no_resume para reescribir).")
                with open(output_path) as f:
                    result = json.load(f)
            else:
                result = run_experiment(
                    instance    = instance,
                    config      = config,
                    adj_matrix  = adj_matrix,
                    baseline    = baseline,
                    n_runs      = n_runs,
                    output_path = output_path,
                    resume      = resume,
                )

            s = result["summary"]
            print(
                f"    ▶ MBF={s['mbf']:.1f} ± {s['mbf_std']:.1f}  "
                f"SR={s['sr']*100:.1f}%  "
                f"AES={s['aes_mean'] if s['aes_mean'] else '---'}"
            )
            all_results.append(result)

    # ── CSV de resumen global ────────────────────────────────────────────────
    build_summary_csv(results_dir)

    elapsed = time.perf_counter() - t_global
    h, rem = divmod(int(elapsed), 3600)
    m, s   = divmod(rem, 60)
    print(f"\n[FIN] Tiempo total: {h}h {m}m {s}s")


if __name__ == "__main__":
    main()