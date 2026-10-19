# problems.py
import numpy as np


def rastrigin(point):
    x, y = point
    return float(
        20 + x**2 + y**2
        - 10 * (np.cos(2 * np.pi * x)
                + np.cos(2 * np.pi * y))
    )


def rosenbrock(point):
    x, y = point
    return float(
        (1 - x)**2 + 100 * (y - x**2)**2
    )

PROBLEMS = {
    "rastrigin": {
        "name": "Rastrigin Function",
        "function": rastrigin,
        "optimum": [0.0, 0.0],
        "minimum": 0.0,
        "bounds": [-5.12, 5.12],
    },
    "rosenbrock": {
        "name": "Rosenbrock Function",
        "function": rosenbrock,
        "optimum": [1.0, 1.0],
        "minimum": 0.0,
        "bounds": [-2.0, 2.0],
    },
}