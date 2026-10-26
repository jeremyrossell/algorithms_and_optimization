import math
import random
import time
import matplotlib.pyplot as plt

# Generate random 2D city coordinates within a 100x100 space
def generate_cities(num_cities: int, seed: int = 42) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    return [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(num_cities)]

# Precompute pairwise Euclidean distances between cities
def compute_distance_matrix(cities: list[tuple[float, float]]) -> list[list[float]]:
    n = len(cities)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = math.dist(cities[i], cities[j])
    return matrix

# Calculate the closed-loop distance of a tour
def tour_length(tour: list[int], dist_matrix: list[list[float]]) -> float:
    n = len(tour)
    return sum(dist_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))

# Generate a random 2-opt neighbor state and calculates cost delta
def random_2opt_successor(tour: list[int], dist_matrix: list[list[float]]) -> tuple[list[int], float]:
    n = len(tour)
    i, k = sorted(random.sample(range(n), 2))

    # we ensure non-adjacent indices for valid 2-opt move
    if i == 0 and k == n - 1:
        return list(tour), 0.0

    a, b = tour[i - 1], tour[i]
    c, d = tour[k], tour[(k + 1) % n]

    delta_cost = (dist_matrix[a][c] + dist_matrix[b][d]) - (dist_matrix[a][b] + dist_matrix[c][d])

    new_tour = tour[:i] + list(reversed(tour[i:k + 1])) + tour[k + 1:]
    return new_tour, delta_cost

# Executes Simulated Annealing optimization based on the SA pseudocode
def simulated_annealing(initial_tour: list[int], dist_matrix: list[list[float]],
                        schedule_fn, total_steps: int = 20_000, seed: int = 42):
    rng = random.Random(seed)
    current_tour = list(initial_tour)
    current_cost = tour_length(current_tour, dist_matrix)

    best_tour = list(current_tour)
    best_cost = current_cost

    cost_history = []

    t0 = time.perf_counter()

    for step in range(1, total_steps + 1):
        T = schedule_fn(step, total_steps)
        if T <= 1e-6:
            break

        next_tour, delta_cost = random_2opt_successor(current_tour, dist_matrix)

        # minimization: delta_E = current_cost - next_cost = -delta_cost
        if delta_cost < 0:
            current_tour = next_tour
            current_cost += delta_cost
            if current_cost < best_cost:
                best_tour = list(current_tour)
                best_cost = current_cost
        else:
            # accept worsening move with Metropolis probability P = exp(-delta_cost / T)
            prob = math.exp(-delta_cost / T)
            if rng.random() < prob:
                current_tour = next_tour
                current_cost += delta_cost

        cost_history.append(current_cost)

    execution_time = time.perf_counter() - t0
    return best_tour, best_cost, execution_time, cost_history


def main():
    num_cities = 50
    total_steps = 30_000
    cities = generate_cities(num_cities, seed=2026)
    dist_matrix = compute_distance_matrix(cities)

    # initial random solution
    init_tour = list(range(num_cities))
    random.Random(101).shuffle(init_tour)
    initial_cost = tour_length(init_tour, dist_matrix)

    T_start = 200.0
    T_end = 0.01

    # Cooling schedules definitions
    def exponential_schedule(step, max_steps):
        alpha = (T_end / T_start) ** (1.0 / max_steps)
        return T_start * (alpha ** step)

    def linear_schedule(step, max_steps):
        return max(T_end, T_start - (T_start - T_end) * (step / max_steps))

    def logarithmic_schedule(step, max_steps):
        c = (T_start / T_end - 1.0) / math.log(1 + max_steps)
        return T_start / (1.0 + c * math.log(1.0 + step))

    schedules = {
        "Exponential": exponential_schedule,
        "Linear": linear_schedule,
        "Logarithmic": logarithmic_schedule
    }

    results = {}

    print(f"=== Simulated Annealing Parameter Tuning ({num_cities} Cities, {total_steps:,} Steps) ===")
    print(f"Initial Random Tour Length: {initial_cost:.2f}\n")
    print(f"{'Cooling Schedule':<20}{'Final Tour Length':<22}{'Execution Time (s)':<20}")
    print("-" * 62)

    for name, sched in schedules.items():
        best_t, best_c, exec_t, history = simulated_annealing(
            init_tour, dist_matrix, sched, total_steps=total_steps, seed=42
        )
        results[name] = {
            "best_cost": best_c,
            "exec_time": exec_t,
            "history": history
        }
        print(f"{name:<20}{best_c:<22.2f}{exec_t:<20.4f}")

    print("-" * 62)

    plt.figure(figsize=(10, 5))
    for name, data in results.items():
        plt.plot(data["history"], label=f"{name} (Final = {data['best_cost']:.1f})", linewidth=1.8)
    plt.axhline(y=initial_cost, color='gray', linestyle='--', alpha=0.7, label='Initial Random State')
    plt.title(r'Simulated Annealing Convergence Curves by Cooling Schedule', fontsize=13)
    plt.xlabel('Iterations (Steps)', fontsize=11)
    plt.ylabel('Tour Cost (Distance)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('sa_convergence_curves.png', dpi=300)
    plt.show()

    # Visualization 2: Execution Time vs. Final Quality
    fig, ax1 = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    costs = [results[m]["best_cost"] for m in names]
    times = [results[m]["exec_time"] for m in names]

    x = range(len(names))
    width = 0.35

    rects1 = ax1.bar([p - width / 2 for p in x], costs, width, label='Final Tour Length', color='tab:blue')
    ax1.set_ylabel('Final Tour Distance (Lower is Better)', color='tab:blue', fontsize=11)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(400, max(costs) + 100)

    ax2 = ax1.twinx()
    rects2 = ax2.bar([p + width / 2 for p in x], times, width, label='Execution Time (s)', color='tab:orange')
    ax2.set_ylabel('Execution Time (seconds)', color='tab:orange', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title(r'Simulated Annealing: Execution Time vs. Solution Quality', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=11)
    ax1.bar_label(rects1, padding=3, fmt='%.1f')
    ax2.bar_label(rects2, padding=3, fmt='%.3fs')

    fig.tight_layout()
    plt.savefig('sa_time_vs_quality.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
