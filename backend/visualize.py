
import numpy as np
import matplotlib.pyplot as plt

from problems import PROBLEMS


def plot_3d(problem_id):
    problem = PROBLEMS[problem_id]
    function = problem["function"]
    lower, upper = problem["bounds"]

    # Create a grid of x and y coordinates
    values = np.linspace(lower, upper, 150)
    X, Y = np.meshgrid(values, values)

    # Evaluate the function at every grid point
    Z = function([X, Y])

    # Create a separate 3D figure
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        X, Y, Z,
        cmap="viridis",
        linewidth=0,
        antialiased=True
    )

    ax.set_title(problem["name"] if "name" in problem else problem_id.title())
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("f(x, y)")

    fig.colorbar(surface, ax=ax, shrink=0.6, label="Function value")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Choose a problem:")
    for i, problem_id in enumerate(PROBLEMS, start=1):
        print(f"{i}. {problem_id.title()}")

    choice = int(input("Your choice: "))
    problem_id = list(PROBLEMS)[choice - 1]

    plot_3d(problem_id)
