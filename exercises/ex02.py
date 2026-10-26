import matplotlib.pyplot as plt

class Item:
    def __init__(self, name: str, weight: int, value: int):
        self.name = name
        self.weight = weight
        self.value = value
        self.ratio = value / weight

    def __repr__(self):
        return f"{self.name}(w={self.weight}, v={self.value}, v/w={self.ratio:.1f})"

# Greedy selection prioritizing items with the lowest weight first
def greedy_lowest_weight(items: list[Item], capacity: int):
    sorted_items = sorted(items, key=lambda item: item.weight)
    selected = []
    total_weight = 0
    total_value = 0

    for item in sorted_items:
        if total_weight + item.weight <= capacity:
            selected.append(item)
            total_weight += item.weight
            total_value += item.value

    return selected, total_weight, total_value

# Greedy selection prioritizing items with the highest value-to-weight ratio first
def greedy_value_weight_ratio(items: list[Item], capacity: int):
    sorted_items = sorted(items, key=lambda item: item.ratio, reverse=True)
    selected = []
    total_weight = 0
    total_value = 0

    for item in sorted_items:
        if total_weight + item.weight <= capacity:
            selected.append(item)
            total_weight += item.weight
            total_value += item.value

    return selected, total_weight, total_value

def main():
    # Knapsack capacity
    capacity = 10

    # Custom counterexample isntance:
    # Item 1: w=2, v=10  (ratio = 5.0)
    # Item 2: w=3, v=12  (ratio = 4.0)
    # Item 3: w=8, v=80  (ratio = 10.0)
    items = [
        Item("Item 1", 2, 10),
        Item("Item 2", 3, 12),
        Item("Item 3", 8, 80)
    ]

    lw_selected, lw_weight, lw_value = greedy_lowest_weight(items, capacity)
    vwr_selected, vwr_weight, vwr_value = greedy_value_weight_ratio(items, capacity)

    lw_names = ", ".join([item.name for item in lw_selected])
    vwr_names = ", ".join([item.name for item in vwr_selected])

    print("=== Small Instance Knapsack Comparison ===")
    print(f"Capacity: {capacity}\n")
    print(f"{'Strategy':<26}{'Items Selected':<22}{'Used Capacity':<16}{'Total Value':<12}")
    print("-" * 76)
    print(f"{'Lowest Weight First':<26}{lw_names:<22}{f'{lw_weight}/{capacity}':<16}{f'${lw_value}':<12}")
    print(f"{'Value/Weight Ratio First':<26}{vwr_names:<22}{f'{vwr_weight}/{capacity}':<16}{f'${vwr_value}':<12}")
    print("-" * 76)

    strategies = ['Lowest Weight First', 'Value/Weight Ratio']
    values = [lw_value, vwr_value]
    used_weights = [lw_weight, vwr_weight]

    x = range(len(strategies))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))

    rects1 = ax1.bar([p - width/2 for p in x], values, width, label='Total Value ($)', color='tab:blue')
    ax1.set_ylabel('Total Value ($)', color='tab:blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    rects2 = ax2.bar([p + width/2 for p in x], used_weights, width, label='Used Capacity', color='tab:orange')
    ax2.set_ylabel(f'Capacity Used (Max = {capacity})', color='tab:orange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylim(0, capacity + 2)

    plt.title(r'Greedy Knapsack Strategies: Lowest Weight vs. Value/Weight Ratio', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=11)
    ax1.bar_label(rects1, padding=3, fmt='$%d')
    ax2.bar_label(rects2, padding=3, fmt='%d units')

    fig.tight_layout()
    plt.savefig('knapsack_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
