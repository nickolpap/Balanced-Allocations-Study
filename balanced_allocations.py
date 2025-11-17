import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

# Define allocation strategies
def one_choice(n, m):
    """
    Standard one-choice allocation: each ball goes to one random bin
    """
    bins = [0] * m
    for i in range(n):
        bin_idx = random.randint(0, m-1)
        bins[bin_idx] += 1
    return bins

def two_choice(n, m):
    """
    Two-choice allocation: choose two random bins, place ball in less loaded one
    """
    bins = [0] * m
    for i in range(n):
        bin1 = random.randint(0, m-1)
        bin2 = random.randint(0, m-1)
        # Choose bin with fewer balls (break ties randomly)
        if bins[bin1] < bins[bin2]:
            bins[bin1] += 1
        elif bins[bin1] > bins[bin2]:
            bins[bin2] += 1
        else:
            # Tie - choose randomly
            if random.random() < 0.5:
                bins[bin1] += 1
            else:
                bins[bin2] += 1
    return bins

def beta_choice(n, m, beta):
    """
    (1+β)-choice: with probability β use two-choice strategy,
    with probability (1-β) use one-choice strategy
    """
    if not (0 <= beta <= 1):
        raise ValueError("Beta must be between 0 and 1")
    
    bins = [0] * m
    for i in range(n):
        if random.random() < beta:
            # Two-choice strategy
            bin1 = random.randint(0, m-1)
            bin2 = random.randint(0, m-1)
            if bins[bin1] <= bins[bin2]:
                bins[bin1] += 1
            else:
                bins[bin2] += 1
        else:
            # One-choice strategy
            bin_idx = random.randint(0, m-1)
            bins[bin_idx] += 1
    return bins


def run_experiment(n, m, strategy, params=None, num_runs=30):
    """
    Run allocation experiment multiple times and return averaged results
    """
    max_loads = []
    gaps = []
    std_devs = []
    
    for _ in range(num_runs):
        if strategy == 'one_choice':
            bins = one_choice(n, m)
        elif strategy == 'two_choice':
            bins = two_choice(n, m)
        elif strategy == 'beta_choice':
            bins = beta_choice(n, m, params['beta'])
        
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
    print("RUNNING COMPREHENSIVE EXPERIMENTS")
    print("=" * 80)
    
    # Experiment parameters
    m = 100  # Number of bins
    n_values = [100, 500, 1000, 2500, 5000, 10000]  # Number of balls to test
    
    print(f"Parameters: m = {m} bins, {len(n_values)} different n values")
    print(f"Number of runs per experiment: 30")
    print()
    
    results_data = []
    
    for n in n_values:
        print(f"--- n = {n} (n/m = {n/m}) ---")
        
        # Run one-choice strategy
        one_result = run_experiment(n, m, 'one_choice')
        print(f"One-choice:    Max Load = {one_result['avg_max_load']:6.2f} ± {one_result['std_max_load']:4.2f}, "
              f"Gap = {one_result['avg_gap']:6.2f}")
        
        # Run two-choice strategy  
        two_result = run_experiment(n, m, 'two_choice')
        print(f"Two-choice:    Max Load = {two_result['avg_max_load']:6.2f} ± {two_result['std_max_load']:4.2f}, "
              f"Gap = {two_result['avg_gap']:6.2f}")
        
        # Run (1+β)-choice strategy with β=0.5
        beta_result = run_experiment(n, m, 'beta_choice', {'beta': 0.5})
        print(f"(1+β)-choice:  Max Load = {beta_result['avg_max_load']:6.2f} ± {beta_result['std_max_load']:4.2f}, "
              f"Gap = {beta_result['avg_gap']:6.2f}")
        
        print()
        
        # Store results for DataFrame
        results_data.extend([
            {'n': n, 'Strategy': 'One-choice', 'Max_Load': one_result['avg_max_load'], 
             'Gap': one_result['avg_gap'], 'Std_Dev': one_result['avg_std_dev']},
            {'n': n, 'Strategy': 'Two-choice', 'Max_Load': two_result['avg_max_load'], 
             'Gap': two_result['avg_gap'], 'Std_Dev': two_result['avg_std_dev']},
            {'n': n, 'Strategy': '(1+β)-choice', 'Max_Load': beta_result['avg_max_load'], 
             'Gap': beta_result['avg_gap'], 'Std_Dev': beta_result['avg_std_dev']}
        ])
    
    # Create DataFrame from results
    df_results = pd.DataFrame(results_data)
    print("All experiments completed successfully!")
    
    # Generate plots
    print("Generating plots for report...")
    
    # Set up the figure with 4 subplots
    plt.figure(figsize=(15, 12))
    
    # Define plot parameters
    strategies = ['One-choice', 'Two-choice', '(1+β)-choice']
    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']
    
    # Plot 1: Gap vs n for all strategies
    plt.subplot(2, 2, 1)
    for i, strategy in enumerate(strategies):
        strategy_data = df_results[df_results['Strategy'] == strategy]
        plt.plot(strategy_data['n'], strategy_data['Gap'], 
                 marker=markers[i], color=colors[i], linewidth=2, 
                 label=strategy, markersize=6)
    
    plt.xlabel('Number of Balls (n)')
    plt.ylabel('Gap (Gₙ)')
    plt.title('Evolution of Gap Gₙ for Different Strategies')
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
    
    # Plot 4: Improvement Ratio (Two-choice / One-choice)
    plt.subplot(2, 2, 4)
    one_data = df_results[df_results['Strategy'] == 'One-choice']
    two_data = df_results[df_results['Strategy'] == 'Two-choice']
    improvement_ratio = one_data['Max_Load'].values / two_data['Max_Load'].values
    
    plt.plot(one_data['n'], improvement_ratio, 'purple', marker='D', 
             linewidth=2, markersize=6, label='Improvement Ratio')
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('Number of Balls (n)')
    plt.ylabel('Improvement Ratio')
    plt.title('Two-choice vs One-choice (Max Load Ratio)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('balanced_allocations_plots.png', dpi=300, bbox_inches='tight')
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