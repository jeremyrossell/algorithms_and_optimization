import itertools
import math
import random
import time
import matplotlib.pyplot as plt


# Metric TSP utilities

def generate_cities(num_cities: int, rng: random.Random) -> list[tuple[float, float]]:
    return [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(num_cities)]

def compute_dist_matrix(points: list[tuple[float, float]]) -> list[list[float]]:
    n = len(points)
    return [[math.dist(points[i], points[j]) for j in range(n)] for i in range(n)]

def tour_length(tour: list[int], dist_matrix: list[list[float]]) -> float:
    n = len(tour)
    return sum(dist_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))


# --- Exact Optimal Tour (Brute Force) --

# Brute force optimal TSP tour fixing city 0 as root (0 + permutations of 1..n-1)
def optimal_tour_brute_force(num_cities: int, dist_matrix: list[list[float]]) -> tuple[list[int], float]:
    best_perm = min(
        ((0,) + p for p in itertools.permutations(range(1, num_cities))),
        key=lambda t: tour_length(t, dist_matrix)
    )
    return list(best_perm), tour_length(best_perm, dist_matrix)


# --- MST 2-Approximation Algorithm ---

# Prim's algorithm starting from city 0; returns adjacency dict {node: children}
def mst_prim(num_cities: int, dist_matrix: list[list[float]]) -> dict[int, list[int]]:
    best = {v: (dist_matrix[0][v], 0) for v in range(1, num_cities)}
    children = {v: [] for v in range(num_cities)}
    while best:
        v = min(best, key=lambda u: best[u][0])
        parent = best.pop(v)[1]
        children[parent].append(v)
        for u in best:
            d = dist_matrix[v][u]
            if d < best[u][0]:
                best[u] = (d, v)
    return children

# Preorder depth-first traversal of the MST (shortcutting repeated visits)
def preorder_walk(children: dict[int, list[int]], root: int = 0) -> list[int]:
    return [root] + [c for child in children[root] for c in preorder_walk(children, child)]

def mst_approximation_tour(num_cities: int, dist_matrix: list[list[float]]) -> tuple[list[int], float]:
    children = mst_prim(num_cities, dist_matrix)
    tour = preorder_walk(children)
    return tour, tour_length(tour, dist_matrix)


# Local Search (2-opt)

def two_opt_local_search(initial_tour: list[int], dist_matrix: list[list[float]]) -> tuple[list[int], float]:
    tour = list(initial_tour)
    n = len(tour)
    improved = True

    while improved:
        improved = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                a, b = tour[i - 1], tour[i]
                c, d = tour[k], tour[(k + 1) % n]
                delta = (dist_matrix[a][c] + dist_matrix[b][d]) - (dist_matrix[a][b] + dist_matrix[c][d])
                if delta < -1e-9:
                    tour[i:k + 1] = reversed(tour[i:k + 1])
                    improved = True
                    break
            if improved:
                break

    return tour, tour_length(tour, dist_matrix)


# Simulated Annealing

def simulated_annealing(initial_tour: list[int], dist_matrix: list[list[float]],
                        steps: int = 5_000, seed: int = 42) -> tuple[list[int], float]:
    rng = random.Random(seed)
    current_tour = list(initial_tour)
    current_cost = tour_length(current_tour, dist_matrix)
    best_tour = list(current_tour)
    best_cost = current_cost

    T_start, T_end = 200.0, 0.01
    alpha = (T_end / T_start) ** (1.0 / steps)
    T = T_start

    for _ in range(steps):
        i, k = sorted(rng.sample(range(len(current_tour)), 2))
        if i == 0 and k == len(current_tour) - 1:
            continue
        a, b = current_tour[i - 1], current_tour[i]
        c, d = current_tour[k], current_tour[(k + 1) % len(current_tour)]
        delta = (dist_matrix[a][c] + dist_matrix[b][d]) - (dist_matrix[a][b] + dist_matrix[c][d])

        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_tour = current_tour[:i] + list(reversed(current_tour[i:k + 1])) + current_tour[k + 1:]
            current_cost += delta
            if current_cost < best_cost:
                best_tour = list(current_tour)
                best_cost = current_cost
        T *= alpha

    return best_tour, best_cost



def run_benchmark():
    num_cities = 9
    num_instances = 100
    master_rng = random.Random(2026)

    mst_ratios = []
    mst_2opt_ratios = []

    costs_opt = []
    costs_mst = []
    costs_mst_2opt = []
    costs_rand_2opt = []
    costs_sa = []

    print(f"Running benchmark across {num_instances} instances of {num_cities} cities...\n")

    for idx in range(num_instances):
        cities = generate_cities(num_cities, master_rng)
        dist_matrix = compute_dist_matrix(cities)

        # 1. Exact optimal (Brute Force)
        _, c_opt = optimal_tour_brute_force(num_cities, dist_matrix)
        costs_opt.append(c_opt)

        # 2. Pure MST approximation
        tour_mst, c_mst = mst_approximation_tour(num_cities, dist_matrix)
        costs_mst.append(c_mst)
        mst_ratios.append(c_mst / c_opt)

        # 3. MST + 2-opt post-processing
        _, c_mst_2opt = two_opt_local_search(tour_mst, dist_matrix)
        costs_mst_2opt.append(c_mst_2opt)
        mst_2opt_ratios.append(c_mst_2opt / c_opt)

        # 4. Random 2-opt local search
        rand_tour = list(range(num_cities))
        master_rng.shuffle(rand_tour)
        _, c_rand_2opt = two_opt_local_search(rand_tour, dist_matrix)
        costs_rand_2opt.append(c_rand_2opt)

        # 5. Simulated Annealing
        _, c_sa = simulated_annealing(rand_tour, dist_matrix, steps=3_000, seed=idx)
        costs_sa.append(c_sa)

    print(f"{'Metric':<28}{'Pure MST':<16}{'MST + 2-opt':<16}{'Theoretical Bound':<18}")
    print("-" * 78)
    print(
        f"{'Mean Ratio (C/C_opt)':<28}{sum(mst_ratios) / num_instances:<16.3f}{sum(mst_2opt_ratios) / num_instances:<16.3f}{'2.000':<18}")
    print(f"{'Max Ratio Observed':<28}{max(mst_ratios):<16.3f}{max(mst_2opt_ratios):<16.3f}{'2.000':<18}")
    print(f"{'Min Ratio Observed':<28}{min(mst_ratios):<16.3f}{min(mst_2opt_ratios):<16.3f}{'1.000':<18}")
    print("-" * 78)

    # Plot 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.hist(mst_ratios, bins=15, alpha=0.7, color='tab:blue', edgecolor='black', label='Pure MST Ratio')
    ax1.hist(mst_2opt_ratios, bins=15, alpha=0.7, color='tab:green', edgecolor='black', label='MST + 2-opt Ratio')
    ax1.axvline(x=2.0, color='tab:red', linestyle='--', linewidth=2.5, label=r'Theoretical Limit ($\rho = 2.0$)')
    ax1.set_title(r'Distribution of Approximation Ratio ($C_{\text{obtained}} / C_{\text{optimal}}$)', fontsize=12)
    ax1.set_xlabel(r'Approximation Ratio', fontsize=11)
    ax1.set_ylabel(r'Frequency (Instances)', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=10)

    # scatter plot of instance ratios
    ax2.scatter(range(num_instances), mst_ratios, color='tab:blue', alpha=0.6, label='Pure MST')
    ax2.scatter(range(num_instances), mst_2opt_ratios, color='tab:green', alpha=0.8, marker='^', label='MST + 2-opt')
    ax2.axhline(y=2.0, color='tab:red', linestyle='--', linewidth=2.5, label=r'Theoretical Bound ($\rho = 2.0$)')
    ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, label=r'Optimal ($\rho = 1.0$)')
    ax2.set_title(r'Instance-by-Instance Ratio Scatter Plot', fontsize=12)
    ax2.set_xlabel(r'Instance Index', fontsize=11)
    ax2.set_ylabel(r'Ratio ($C / C_{\text{optimal}}$)', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('mst_approximation_ratios.png', dpi=300)
    plt.show()

    # plot 2
    plt.figure(figsize=(10, 5))
    data = [costs_opt, costs_mst, costs_mst_2opt, costs_rand_2opt, costs_sa]
    labels = ['Optimal\n(Brute)', 'Pure MST\n(2-Approx)', 'MST +\n2-opt', 'Random +\n2-opt', 'Simulated\nAnnealing']

    plt.boxplot(data, patch_artist=True,
                boxprops=dict(facecolor='lightcyan', color='darkcyan'),
                medianprops=dict(color='crimson', linewidth=2))
    plt.xticks(range(1, len(labels) + 1), labels, fontsize=10)
    plt.title(r'Tour Cost Comparison Across Optimization Algorithms (100 Instances, N=9)', fontsize=13)
    plt.ylabel(r'Total Tour Distance', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('tsp_methods_comparison.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    run_benchmark()
