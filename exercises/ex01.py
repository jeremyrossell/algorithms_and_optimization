import multiprocessing as mp
import random
import time
import matplotlib.pyplot as plt

# Monte Carlo estimation of Pi using a single process
def estimate_pi(seed: int, samples: int = 5_000_000) -> float:
    rng = random.Random(seed)
    inside = sum(rng.random()**2 + rng.random()**2 <= 1.0 for _ in range(samples))
    return 4 * inside / samples

def run_benchmark():
    # use 16 independent seed tasks to ensure equal work distribution across 1, 2, 4, and 8 workers
    num_tasks = 16
    seeds = list(range(num_tasks))
    process_counts = [1, 2, 4, 8]
    execution_times = []
    speedups = []

    print(f"Running benchmark with {num_tasks} tasks ({5_000_000:,} samples each)...\n")
    print(f"{'Processes':<12}{'Wall-Clock (s)':<18}{'Speedup':<14}{'Ideal Speedup':<14}")
    print("-" * 58)

    t1_base = None

    for p in process_counts:
        t0 = time.perf_counter()
        with mp.Pool(processes=p) as pool:
            results = pool.map(estimate_pi, seeds)
        t_wall = time.perf_counter() - t0

        if p == 1:
            t1_base = t_wall

        speedup = t1_base / t_wall
        execution_times.append(t_wall)
        speedups.append(speedup)

        sp_str = f"{speedup:.2f}x"
        ideal_str = f"{float(p):.2f}x"
        print(f"{p:<12}{t_wall:<18.3f}{sp_str:<14}{ideal_str:<14}")

    plt.figure(figsize=(8, 5))
    plt.plot(process_counts, speedups, 'o-', label='Actual Speedup', color='tab:blue', linewidth=2, markersize=8)
    plt.plot(process_counts, process_counts, '--', label='Ideal Linear Speedup ($S = P$)', color='tab:red', linewidth=2)
    plt.title(r'Parallel Speedup: Monte Carlo $\pi$ Estimation', fontsize=14)
    plt.xlabel('Number of Processes ($P$)', fontsize=12)
    plt.ylabel(r'Speedup ($S_P = T_1 / T_P$)', fontsize=12)
    plt.xticks(process_counts)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('parallel_speedup.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_benchmark()
