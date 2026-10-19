
import numpy as np
import matplotlib.pyplot as plt

from algorithms import (
    gradient_descent,
    simulated_annealing,
    genetic_algorithm,
)

ALGORITHMS = {
    "Gradient Descent": gradient_descent,
    "Simulated Annealing": simulated_annealing,
    "Genetic Algorithm": genetic_algorithm,
}

COLORS = {
    "Gradient Descent": "dodgerblue",
    "Simulated Annealing": "darkorange",
    "Genetic Algorithm": "limegreen",
}


def compare_3d(problem, iterations=500, seed=42):
    function = problem["function"]
    lower, upper = problem["bounds"]

    values = np.linspace(lower, upper, 100)
    X, Y = np.meshgrid(values, values)
    Z = function([X, Y])

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.5)

    optimum = problem["optimum"]
    ax.scatter(
        optimum[0], optimum[1], problem["minimum"],
        color="gold", marker="*", s=250,
        edgecolors="black", label="Known global minimum",
    )

    
    for name, algorithm in ALGORITHMS.items():
        print(f"\nRunning {name}...")

        result = algorithm(
            function=function,
            bounds=problem["bounds"],
            iterations=iterations,
            seed=seed,
        )

        path = np.asarray(result["path"], dtype=float)

        print(f"\n{name} - Path Coordinates")
        print("-" * 65)
        print(f"{'Step':<8}{'X':>15}{'Y':>15}{'f(x, y)':>20}")

        for step, point in enumerate(path):
            x, y = point
            z = float(function(point))

            print(f"{step:<8}{x:>15.6f}{y:>15.6f}{z:>20.6f}")

        print(f"\nBest position: {result['best_position']}")
        print(f"Best score:    {result['best_score']:.8f}")
        print(f"Function evaluations: {result['function_evaluations']}")


        result = algorithm(
            function=function,
            bounds=problem["bounds"],
            iterations=iterations,
            seed=seed,
        )

        path = np.asarray(result["path"], dtype=float)

        if path.ndim == 2 and len(path) > 0:
            heights = np.asarray([
                float(function(point)) for point in path
            ])

            ax.plot(
                path[:, 0], path[:, 1], heights,
                color=COLORS[name], linewidth=2,
                marker="o", markersize=2, label=name,
            )

        best = result["best_position"]
        ax.scatter(
            best[0], best[1], result["best_score"],
            color=COLORS[name], marker="X", s=90,
            edgecolors="black",
        )

        print(
            f"{name}: score={result['best_score']:.8f}, "
            f"time={result['elapsed_ms']:.2f} ms"
        )

    ax.set_title("Optimisation Arena — 3D Algorithm Comparison")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("f(x, y)")
    ax.legend()

    plt.tight_layout()
    plt.show()
