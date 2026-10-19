
from problems import PROBLEMS
from algorithms import (
    gradient_descent,
    simulated_annealing,
    genetic_algorithm,
)
import pyfiglet


ALGORITHMS = {
    "1": ("Gradient Descent", gradient_descent),
    "2": ("Simulated Annealing", simulated_annealing),
    "3": ("Genetic Algorithm", genetic_algorithm),
}


def main():
    print("\n" + "=" * 45)
    slant_art = pyfiglet.figlet_format("OPTIMIZATION ARENA", font="slant")
    print(slant_art)
    print("=" * 45)

    print("\nChoose a mathematical problem:")

    problem_ids = list(PROBLEMS)

    for number, problem_id in enumerate(problem_ids, start=1):
        print(f"{number}. {PROBLEMS[problem_id]['name']}")  

    choice = input("\nProblem number: ").strip()

    if not choice.isdigit() or not 1 <= int(choice) <= len(problem_ids):
        print("Invalid problem selection.")
        return

    problem_id = problem_ids[int(choice) - 1]
    problem = PROBLEMS[problem_id]

    print("\nChoose algorithms:")
    for number, (name, _) in ALGORITHMS.items():
        print(f"{number}. {name}")
    print("4. Compare all algorithms")

    selection = input("\nYour choice: ").strip()

    if selection == "4":
        selected_algorithms = list(ALGORITHMS.values())
    elif selection in ALGORITHMS:
        selected_algorithms = [ALGORITHMS[selection]]
    else:
        print("Invalid algorithm selection.")
        return

    try:
        iterations = int(
            input("Number of iterations (default 500): ") or "500"
        )
        if not 1 <= iterations <= 10000:
            raise ValueError
    except ValueError:
        print("Enter a whole number between 1 and 10000.")
        return

    print(f"\nProblem: {problem['name']}")
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

    results.sort(key=lambda item: item["best_score"])

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





#TO RUN 
# python -u "c:\Users\pc\Desktop\pilot\backend\main.py"