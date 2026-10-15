from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from deap import creator, tools

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.graph_chordalizer as gc_mod
from src.graph_chordalizer import GraphChordalizer


def _cycle_adjacency(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        adj[i, (i + 1) % n] = 1
        adj[(i + 1) % n, i] = 1
    return adj


def _force_identical_indexes(ea: GraphChordalizer) -> None:
    """Re-registra índices fijos; DEAP captura la alias al registrar ``individual``."""
    ea.toolbox.register("indexes", lambda: list(range(ea.num_vertex)))
    ea.toolbox.register("individual", tools.initIterate, creator.Individual, ea.toolbox.indexes)
    ea.toolbox.register("population", tools.initRepeat, list, ea.toolbox.individual)


def _count_fillin_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    real_count: Callable = gc_mod.count_fillin
    state = {"n": 0}

    def _wrapped(adj_matrix, ordering, *, validate: bool = True) -> int:
        state["n"] += 1
        return real_count(adj_matrix, ordering, validate=validate)

    monkeypatch.setattr(gc_mod, "count_fillin", _wrapped)
    return state


def test_duplicate_genotypes_call_count_fillin_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _count_fillin_calls(monkeypatch)
    ea = GraphChordalizer(_cycle_adjacency(6))
    ea._fitness_cache.clear()

    perm = [0, 1, 2, 3, 4, 5]
    ind_a = creator.Individual(perm)
    ind_b = creator.Individual(list(perm))

    new_evals = ea._assign_fitness([ind_a, ind_b], remaining=None)

    assert new_evals == 1
    assert calls["n"] == 1
    assert ind_a.fitness.valid and ind_b.fitness.valid
    assert ind_a.fitness.values == ind_b.fitness.values


def test_remaining_zero_skips_cache_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _count_fillin_calls(monkeypatch)
    ea = GraphChordalizer(_cycle_adjacency(6))
    ea._fitness_cache.clear()
    ind = creator.Individual([5, 4, 3, 2, 1, 0])

    new_evals = ea._assign_fitness([ind], remaining=0)

    assert new_evals == 0
    assert calls["n"] == 0
    assert not ind.fitness.valid


def test_run_ea_identical_population_counts_one_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _count_fillin_calls(monkeypatch)
    ea = GraphChordalizer(_cycle_adjacency(6))
    _force_identical_indexes(ea)

    random.seed(0)
    np.random.seed(0)
    _best, logbook = ea.run_ea(
        num_generations=20,
        population_size=8,
        cx_prob=0.0,
        mut_prob=0.0,
        max_evaluations=50,
        use_multiprocessing=False,
    )

    assert calls["n"] == 1
    assert logbook[-1]["evals"] == 1
    assert logbook.select("evals")[-1] <= 50


def test_run_ea_budget_counts_only_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _count_fillin_calls(monkeypatch)
    ea = GraphChordalizer(_cycle_adjacency(6))
    budget = 5

    random.seed(42)
    np.random.seed(42)
    _best, logbook = ea.run_ea(
        num_generations=100,
        population_size=4,
        cx_prob=0.0,
        mut_prob=1.0,
        max_evaluations=budget,
        use_multiprocessing=False,
    )

    used = int(logbook[-1]["evals"])
    assert used <= budget
    assert calls["n"] == used
    assert used == budget


def test_run_ea_resets_cache_between_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _count_fillin_calls(monkeypatch)
    ea = GraphChordalizer(_cycle_adjacency(6))
    _force_identical_indexes(ea)

    kwargs = dict(
        num_generations=1,
        population_size=4,
        cx_prob=0.0,
        mut_prob=0.0,
        max_evaluations=10,
        use_multiprocessing=False,
    )
    ea.run_ea(**kwargs)
    first_calls = calls["n"]
    ea.run_ea(**kwargs)

    assert first_calls == 1
    assert calls["n"] == 2
