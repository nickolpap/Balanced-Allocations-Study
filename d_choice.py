import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

def d_choice(n, m, d):
    """
    d-choice allocation: choose d random bins, place ball in the least loaded one
    """
    bins = [0] * m
    for i in range(n):
        # Choose d random bins
        chosen_bins = [random.randint(0, m-1) for _ in range(d)]
        # Find the best bin
        min_load = bins[chosen_bins[0]]
        best_bin = chosen_bins[0]
        
        for bin_idx in chosen_bins[1:]:
            if bins[bin_idx] < min_load:
                min_load = bins[bin_idx]
                best_bin = bin_idx
            elif bins[bin_idx] == min_load:
                if random.random() < 0.5:
                    best_bin = bin_idx
        
        # Place the ball in the selected bin
        bins[best_bin] += 1
    
    return bins

def run_experiment(n, m, strategy, params=None, num_runs=30):
    """
    Run allocation experiment multiple times and return averaged results
    """
    max_loads = []
    gaps = []
    std_devs = []
    
    for _ in range(num_runs):
        if strategy == 'd_choice':
            bins = d_choice(n, m, params['d'])
        
        max_load = max(bins)
        gap = max_load - n/m  # Gₙ = max load - n/m
        std_dev = np.std(bins)
        
        max_loads.append(max_load)
        gaps.append(gap)
        std_devs.append(std_dev)
    
    # Return averages and statistics
    return {
        'avg_max_load': np.mean(max_loads),
        'std_max_load': np.std(max_loads),
        'avg_gap': np.mean(gaps),
        'std_gap': np.std(gaps),
        'avg_std_dev': np.mean(std_devs)
    }

# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("D-CHOICE STRATEGIES EXPERIMENT")
    print("=" * 80)
    
    # Experiment parameters
    m = 100  # Number of bins
    n_values = [100, 500, 1000, 2500, 5000, 10000]  # Number of balls to test
    d_values = [1, 2, 3, 4, 5]  # Different d values to test
    
    print(f"Parameters: m = {m} bins, {len(n_values)} different n values")
    print(f"Testing d values: {d_values}")
    print(f"Number of runs per experiment: 30")
    print()
    
    results_data = []
    
    for n in n_values:
        print(f"--- n = {n} (n/m = {n/m}) ---")
        
        for d in d_values:
            # Run d-choice strategy for each d value
            d_result = run_experiment(n, m, 'd_choice', {'d': d})
            print(f"d-choice (d={d}): Max Load = {d_result['avg_max_load']:6.2f} ± {d_result['std_max_load']:4.2f}, "
                  f"Gap = {d_result['avg_gap']:6.2f}")
            
            # Store results for DataFrame
            results_data.append({
                'n': n, 
                'Strategy': f'd-choice (d={d})', 
                'Max_Load': d_result['avg_max_load'], 
                'Gap': d_result['avg_gap'], 
                'Std_Dev': d_result['avg_std_dev']
            })
        
        print()
    
    # Create DataFrame from results
    df_results = pd.DataFrame(results_data)
    print("All experiments completed successfully!")
    
    # Generate plots
    print("Generating plots for report...")
    
    # Set up the figure with 3 subplots (without the 4th plot)
    plt.figure(figsize=(15, 10))
    
    # Define plot parameters
    strategies = [f'd-choice (d={d})' for d in d_values]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    # Plot 1: Gap vs n for all strategies
    plt.subplot(2, 2, 1)
    for i, strategy in enumerate(strategies):
        strategy_data = df_results[df_results['Strategy'] == strategy]
        plt.plot(strategy_data['n'], strategy_data['Gap'], 
                 marker=markers[i], color=colors[i], linewidth=2, 
                 label=strategy, markersize=6)
    
    plt.xlabel('Number of Balls (n)')
    plt.ylabel('Gap (Gₙ)')
    plt.title('Evolution of Gap Gₙ for Different d Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    
    # Plot 2: Maximum Load vs n
    plt.subplot(2, 2, 2)
    for i, strategy in enumerate(strategies):
        strategy_data = df_results[df_results['Strategy'] == strategy]
        plt.plot(strategy_data['n'], strategy_data['Max_Load'], 
                 marker=markers[i], color=colors[i], linewidth=2, 
                 label=strategy, markersize=6)
    
    plt.xlabel('Number of Balls (n)')
    plt.ylabel('Maximum Load')
    plt.title('Maximum Load vs Number of Balls')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Standard Deviation vs n
    plt.subplot(2, 2, 3)
    for i, strategy in enumerate(strategies):
        strategy_data = df_results[df_results['Strategy'] == strategy]
        plt.plot(strategy_data['n'], strategy_data['Std_Dev'], 
                 marker=markers[i], color=colors[i], linewidth=2, 
                 label=strategy, markersize=6)
    
    plt.xlabel('Number of Balls (n)')
    plt.ylabel('Standard Deviation')
    plt.title('Load Distribution Standard Deviation vs n')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('d_choice_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("All plots generated successfully!")
    
    # Display results table
    print("=" * 70)
    print("SUMMARY RESULTS TABLE")
    print("=" * 70)
    
    # Create pivot table for better organization
    pivot_df = df_results.pivot_table(
        index='n', 
        columns='Strategy', 
        values=['Max_Load', 'Gap', 'Std_Dev']
    )
    
    # Display formatted table
    with pd.option_context('display.float_format', '{:.2f}'.format):
        print(pivot_df.round(2))
    
    print("PROGRAM COMPLETED")