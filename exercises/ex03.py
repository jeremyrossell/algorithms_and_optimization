import math
import random
import matplotlib.pyplot as plt

# Generate random 2D city coordinates within a 100x100 grid
def generate_cities(num_cities: int, seed: int = 42) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    return [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(num_cities)]

# Precompute Euclidean distances between all pairs of cities
def compute_distance_matrix(cities: list[tuple[float, float]]) -> list[list[float]]:
    n = len(cities)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = math.dist(cities[i], cities[j])
    return matrix

# Calculate the total closed-loop length of a tour
def tour_length(tour: list[int], dist_matrix: list[list[float]]) -> float:
    n = len(tour)
    return sum(dist_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))

# Generates a constructive Nearest Neighbor tour starting from a specific node
def nearest_neighbor_tour(num_cities: int, dist_matrix: list[list[float]], start_node: int) -> list[int]:
    unvisited = set(range(num_cities))
    unvisited.remove(start_node)
    tour = [start_node]
    current = start_node

    while unvisited:
        next_city = min(unvisited, key=lambda city: dist_matrix[current][city])
        unvisited.remove(next_city)
        tour.append(next_city)
        current = next_city

    return tour

# Apply 2-opt local search until no further edge-uncrossing move improves total length
def two_opt_local_search(initial_tour: list[int], dist_matrix: list[list[float]]) -> tuple[list[int], int, float]:
    tour = list(initial_tour)
    n = len(tour)
    improvements = 0
    improved = True

    while improved:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                a, b = tour[i - 1], tour[i]
                c, d = tour[k], tour[(k + 1) % n]

                # Delta distance if subsegment tour[i:k+1] is reversed
                delta = (dist_matrix[a][c] + dist_matrix[b][d]) - (dist_matrix[a][b] + dist_matrix[c][d])

                if delta < -1e-9:  # strict improvement
                    tour[i:k + 1] = reversed(tour[i:k + 1])
                    improvements += 1
                    improved = True
                    break
            if improved:
                break

    return tour, improvements, tour_length(tour, dist_matrix)


def run_experiment():
    num_cities = 50
    num_trials = 50
    cities = generate_cities(num_cities, seed=2026)
    dist_matrix = compute_distance_matrix(cities)

    random_improvements = []
    random_final_lengths = []

    nn_improvements = []
    nn_final_lengths = []

    for trial in range(num_trials):
        rng = random.Random(trial)

        # 1. Random start
        rand_start = list(range(num_cities))
        rng.shuffle(rand_start)
        _, rand_imp, rand_len = two_opt_local_search(rand_start, dist_matrix)
        random_improvements.append(rand_imp)
        random_final_lengths.append(rand_len)

        # 2. Nearest Neighbor start (random start node)
        start_city = rng.randint(0, num_cities - 1)
        nn_start = nearest_neighbor_tour(num_cities, dist_matrix, start_city)
        _, nn_imp, nn_len = two_opt_local_search(nn_start, dist_matrix)
        nn_improvements.append(nn_imp)
        nn_final_lengths.append(nn_len)

    avg_rand_imp = sum(random_improvements) / num_trials
    avg_rand_len = sum(random_final_lengths) / num_trials
    avg_nn_imp = sum(nn_improvements) / num_trials
    avg_nn_len = sum(nn_final_lengths) / num_trials

    print(f"=== 2-Opt Local Search Benchmark ({num_cities} Cities, {num_trials} Trials) ===")
    print(f"{'Initialization Strategy':<28}{'Avg. Improvements':<20}{'Avg. Final Tour Length':<24}")
    print("-" * 72)
    print(f"{'Random Initial Tour':<28}{avg_rand_imp:<20.2f}{avg_rand_len:<24.2f}")
    print(f"{'Nearest Neighbor Tour':<28}{avg_nn_imp:<20.2f}{avg_nn_len:<24.2f}")
    print("-" * 72)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Improvements count
    ax1.boxplot([random_improvements, nn_improvements], patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='darkblue', linewidth=2))
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['Random Start', 'Nearest Neighbor'])
    ax1.set_title(r'Number of 2-Opt Improvements to Converge', fontsize=12)
    ax1.set_ylabel(r'Number of 2-Opt Swaps', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Final tour length (Quality)
    ax2.boxplot([random_final_lengths, nn_final_lengths], patch_artist=True,
                boxprops=dict(facecolor='lightgreen', color='green'),
                medianprops=dict(color='darkgreen', linewidth=2))
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['Random Start', 'Nearest Neighbor'])
    ax2.set_title(r'Final Tour Quality (Length)', fontsize=12)
    ax2.set_ylabel(r'Total Tour Distance', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('local_search_comparison.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    run_experiment()
