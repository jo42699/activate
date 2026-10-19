
import time
import numpy as np


def numerical_gradient(function, point, epsilon=1e-6):
    """Estimate the gradient using central finite differences."""
    point = np.asarray(point, dtype=float)
    gradient = np.zeros_like(point)

    for i in range(len(point)):
        plus = point.copy()
        minus = point.copy()
        plus[i] += epsilon
        minus[i] -= epsilon

        gradient[i] = (
            function(plus) - function(minus)
        ) / (2 * epsilon)

    return gradient


def make_result(
    algorithm,
    best_position,
    best_score,
    iterations,
    evaluations,
    start_time,
    history,
    path,
):
    return {
        "algorithm": algorithm,
        "best_position": np.asarray(best_position).tolist(),
        "best_score": float(best_score),
        "iterations": int(iterations),
        "function_evaluations": int(evaluations),
        "elapsed_ms": round(
            (time.perf_counter() - start_time) * 1000, 3
        ),
        "convergence_history": [float(v) for v in history],
        "path": [np.asarray(p).tolist() for p in path],
    }


def gradient_descent(
    function,
    bounds,
    iterations=1000,
    seed=42,
    learning_rate=0.01,
):
    rng = np.random.default_rng(seed)
    low, high = bounds
    point = rng.uniform(low, high, size=2)

    start_time = time.perf_counter()
    evaluations = 0
    score = function(point)
    evaluations += 1

    best_position = point.copy()
    best_score = score
    history = [best_score]
    path = [best_position.copy()]

    for _ in range(iterations):
        gradient = numerical_gradient(function, point)
        evaluations += 4  # Two dimensions, two evaluations each.

        candidate = np.clip(
            point - learning_rate * gradient, low, high
        )
        candidate_score = function(candidate)
        evaluations += 1

        point = candidate

        if candidate_score < best_score:
            best_score = candidate_score
            best_position = candidate.copy()

        history.append(best_score)
        path.append(best_position.copy())

        if np.linalg.norm(gradient) < 1e-8:
            break

    return make_result(
        "gradient_descent", best_position, best_score,
        len(history) - 1, evaluations, start_time, history, path
    )


def simulated_annealing(
    function,
    bounds,
    iterations=1000,
    seed=42,
    initial_temperature=10.0,
    cooling_rate=0.995,
):
    rng = np.random.default_rng(seed)
    low, high = bounds
    point = rng.uniform(low, high, size=2)

    start_time = time.perf_counter()
    evaluations = 1
    score = function(point)

    best_position = point.copy()
    best_score = score
    temperature = initial_temperature
    history = [best_score]
    path = [best_position.copy()]

    for _ in range(iterations):
        candidate = np.clip(
            point + rng.normal(0, 0.5, size=2),
            low,
            high,
        )
        candidate_score = function(candidate)
        evaluations += 1

        delta = candidate_score - score

        if (
            delta < 0
            or rng.random() < np.exp(
                -delta / max(temperature, 1e-12)
            )
        ):
            point = candidate
            score = candidate_score

        if score < best_score:
            best_score = score
            best_position = point.copy()

        history.append(best_score)
        path.append(best_position.copy())
        temperature *= cooling_rate

    return make_result(
        "simulated_annealing", best_position, best_score,
        iterations, evaluations, start_time, history, path
    )


def genetic_algorithm(
    function,
    bounds,
    iterations=300,
    seed=42,
    population_size=50,
    mutation_probability=0.1,
):
    if population_size < 4:
        raise ValueError("population_size must be at least 4")

    if not 0 <= mutation_probability <= 1:
        raise ValueError("mutation_probability must be between 0 and 1")

    rng = np.random.default_rng(seed)
    low, high = bounds
    population = rng.uniform(
        low, high, size=(population_size, 2)
    )

    start_time = time.perf_counter()
    evaluations = population_size
    scores = np.array([function(p) for p in population])

    best_index = int(np.argmin(scores))
    best_position = population[best_index].copy()
    best_score = float(scores[best_index])
    history = [best_score]
    path = [best_position.copy()]

    for _ in range(iterations):
        # Keep the best individual (elitism).
        new_population = [best_position.copy()]

        while len(new_population) < population_size:
            # Tournament selection.
            candidates = rng.choice(
                population_size, size=3, replace=False
            )
            parent1 = population[
                candidates[np.argmin(scores[candidates])]
            ]

            candidates = rng.choice(
                population_size, size=3, replace=False
            )
            parent2 = population[
                candidates[np.argmin(scores[candidates])]
            ]

            # Arithmetic crossover.
            weight = rng.random()
            child = weight * parent1 + (1 - weight) * parent2

            # Gaussian mutation.
            for j in range(2):
                if rng.random() < mutation_probability:
                    child[j] += rng.normal(0, 0.1 * (high - low))

            child = np.clip(child, low, high)
            new_population.append(child)

        population = np.asarray(new_population)
        scores = np.array([function(p) for p in population])
        evaluations += population_size

        generation_best = int(np.argmin(scores))
        if scores[generation_best] < best_score:
            best_score = float(scores[generation_best])
            best_position = population[generation_best].copy()

        history.append(best_score)
        path.append(best_position.copy())

    return make_result(
        "genetic_algorithm", best_position, best_score,
        iterations, evaluations, start_time, history, path
    )
