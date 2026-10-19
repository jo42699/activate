
from problems import PROBLEMS
from algorithms import (
    gradient_descent,
    simulated_annealing,
    genetic_algorithm,
)
from visualize import compare_3d
import pyfiglet


ALGORITHMS = {
    "1": ("Gradient Descent", gradient_descent),
    "2": ("Simulated Annealing", simulated_annealing),
    "3": ("Genetic Algorithm", genetic_algorithm),
}


def main():
    print("\n" + "=" * 45)

    slant_art = pyfiglet.figlet_format(
        "OPTIMIZATION ARENA",
        font="slant",
    )
    print(slant_art)
    print("=" * 45)

    # Choose the mathematical problem
    print("\nChoose a mathematical problem:")

    problem_ids = list(PROBLEMS)

    for number, problem_id in enumerate(problem_ids, start=1):
        problem_name = PROBLEMS[problem_id].get(
            "name",
            problem_id.title(),
        )
        print(f"{number}. {problem_name}")

    choice = input("\nProblem number: ").strip()

    if not choice.isdigit() or not 1 <= int(choice) <= len(problem_ids):
        print("Invalid problem selection.")
        return

    problem_id = problem_ids[int(choice) - 1]
    problem = PROBLEMS[problem_id]

    # Choose the algorithm or comparison mode
    print("\nChoose algorithms:")

    for number, (name, _) in ALGORITHMS.items():
        print(f"{number}. {name}")

    print("4. Compare all algorithms (terminal)")
    print("5. Compare all algorithms + 3D plot")

    selection = input("\nYour choice: ").strip()

    show_plot = False

    if selection == "4":
        selected_algorithms = list(ALGORITHMS.values())

    elif selection == "5":
        selected_algorithms = list(ALGORITHMS.values())
        show_plot = True

    elif selection in ALGORITHMS:
        selected_algorithms = [ALGORITHMS[selection]]

    else:
        print("Invalid algorithm selection.")
        return

    # Choose iteration count
    try:
        iterations = int(
            input("Number of iterations (default 500): ") or "500"
        )

        if not 1 <= iterations <= 10000:
            raise ValueError

    except ValueError:
        print("Enter a whole number between 1 and 10000.")
        return

    # Run the 3D comparison when option 5 is selected
    if show_plot:
        print(f"\nProblem: {problem.get('name', problem_id.title())}")
        print("Opening the 3D comparison plot...")

        compare_3d(
            problem,
            iterations=iterations,
            seed=42,
        )
        return

    # Normal terminal execution
    print(f"\nProblem: {problem.get('name', problem_id.title())}")
    print(f"Known minimum: {problem['minimum']}")
    print("\nRunning algorithms...\n")

    results = []

    for name, algorithm in selected_algorithms:
        print(f"Running {name}...")

        result = algorithm(
            function=problem["function"],
            bounds=problem["bounds"],
            iterations=iterations,
            seed=42,
        )

        results.append(result)

    # Sort by best score
    results.sort(key=lambda item: item["best_score"])

    # Display results table
    print("\n" + "=" * 78)

    print(
        f"{'Algorithm':<24}"
        f"{'Best score':>15}"
        f"{'Time (ms)':>14}"
        f"{'Evaluations':>15}"
    )

    print("-" * 78)

    for result in results:
        print(
            f"{result['algorithm']:<24}"
            f"{result['best_score']:>15.6f}"
            f"{result['elapsed_ms']:>14.3f}"
            f"{result['function_evaluations']:>15}"
        )

    # Display the best result from this run
    best = results[0]

    print("\nBest result in this run:")
    print(f"Algorithm: {best['algorithm']}")
    print(f"Position:  {best['best_position']}")
    print(f"Score:     {best['best_score']:.8f}")
    print(f"Reference minimum: {problem['minimum']:.8f}")

    print(
        "\nNote: this is one run with one seed, "
        "not proof that an algorithm is generally superior."
    )


if __name__ == "__main__":
    main()
