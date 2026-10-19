
import pytest

from problems import PROBLEMS
from algorithms import (
    gradient_descent,
    simulated_annealing,
    genetic_algorithm,
)


ALGORITHMS = [
    gradient_descent,
    simulated_annealing,
    genetic_algorithm,
]


@pytest.mark.parametrize("problem_id", list(PROBLEMS))
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_algorithm_returns_valid_result(problem_id, algorithm):
    problem = PROBLEMS[problem_id]

    result = algorithm(
        function=problem["function"],
        bounds=problem["bounds"],
        iterations=20,
        seed=42,
    )

    assert result["algorithm"]
    assert len(result["best_position"]) == 2
    assert result["best_score"] >= 0
    assert result["function_evaluations"] > 0
    assert result["elapsed_ms"] >= 0


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_best_score_history_never_increases(algorithm):
    problem = PROBLEMS["rastrigin"]

    result = algorithm(
        function=problem["function"],
        bounds=problem["bounds"],
        iterations=30,
        seed=42,
    )

    history = result["convergence_history"]

    assert len(history) > 0

    for previous, current in zip(history, history[1:]):
        assert current <= previous + 1e-10


def test_genetic_algorithm_rejects_small_population():
    with pytest.raises(ValueError):
        genetic_algorithm(
            function=PROBLEMS["sphere"]["function"],
            bounds=PROBLEMS["sphere"]["bounds"],
            population_size=3,
        )
