import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

# Batched allocation strategies
def batched_one_choice(n, m, b):
    """
    One-choice allocation in batched setting
    """
    bins = [0] * m
    for batch_start in range(0, n, b):
        batch_end = min(batch_start + b, n)
        # Use bin loads from START of batch for all decisions in this batch
        current_bins = bins.copy()
        
        for i in range(batch_start, batch_end):
            bin_idx = random.randint(0, m-1)
            # Update actual bins (but decisions based on current_bins)
            bins[bin_idx] += 1
    return bins

def batched_two_choice(n, m, b):
    """
    Two-choice allocation in batched setting
    """
    bins = [0] * m
    for batch_start in range(0, n, b):
        batch_end = min(batch_start + b, n)
        # Use bin loads from START of batch for all decisions in this batch
        current_bins = bins.copy()
        
        for i in range(batch_start, batch_end):
            bin1 = random.randint(0, m-1)
            bin2 = random.randint(0, m-1)
            # Choose based on loads at beginning of batch
            if current_bins[bin1] <= current_bins[bin2]:
                bins[bin1] += 1
            else:
                bins[bin2] += 1
    return bins

def batched_beta_choice(n, m, b, beta):
    """
    (1+β)-choice in batched setting
    """
    bins = [0] * m
    for batch_start in range(0, n, b):
        batch_end = min(batch_start + b, n)
        current_bins = bins.copy()
        
        for i in range(batch_start, batch_end):
            if random.random() < beta:
                # Two-choice strategy
                bin1 = random.randint(0, m-1)
                bin2 = random.randint(0, m-1)
                if current_bins[bin1] <= current_bins[bin2]:
                    bins[bin1] += 1
                else:
                    bins[bin2] += 1
            else:
                # One-choice strategy
                bin_idx = random.randint(0, m-1)
                bins[bin_idx] += 1
    return bins

def run_batched_experiment(n, m, b, strategy, params=None, num_runs=30):
    """
    Run batched allocation experiment multiple times
    """
    max_loads = []
    gaps = []
    std_devs = []
    
    for _ in range(num_runs):
        if strategy == 'batched_one_choice':
            bins = batched_one_choice(n, m, b)
        elif strategy == 'batched_two_choice':
            bins = batched_two_choice(n, m, b)
        elif strategy == 'batched_beta_choice':
            bins = batched_beta_choice(n, m, b, params['beta'])
        
        max_load = max(bins)
        gap = max_load - n/m
        std_dev = np.std(bins)
        
        max_loads.append(max_load)
        gaps.append(gap)
        std_devs.append(std_dev)
    
    return {
        'avg_max_load': np.mean(max_loads),
        'std_max_load': np.std(max_loads),
        'avg_gap': np.mean(gaps),
        'std_gap': np.std(gaps),
        'avg_std_dev': np.mean(std_devs)
    }

# Main execution for batched experiments
if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING BATCHED ALLOCATION EXPERIMENTS")
    print("=" * 80)
    
    m = 100
    m_squared = m ** 2  # n = m² = 10000
    b_values = [100, 200, 500, 1000, 2000, 5000, 7000]
    
    results_data = []
    
    for b in b_values:
        print(f"\n--- Batch size b = {b} ---")
        
        # Calculate lambda values according to: λ = m²/b
        max_lambda = m_squared // b
        # Test different lambda values: λ = 1, 2, 3, 5, 10, ... up to max_lambda
        lambda_values = [1, 2, 3, 5, 10]
        lambda_values = [lam for lam in lambda_values if lam <= max_lambda]
        
        for lam in lambda_values:
            n = lam * b  # n = λ·b
            if n > m_squared:
                continue
                
            print(f"  n = {n} (λ = {lam})")
            
            # Run batched strategies
            one_result = run_batched_experiment(n, m, b, 'batched_one_choice')
            two_result = run_batched_experiment(n, m, b, 'batched_two_choice')
            beta_result = run_batched_experiment(n, m, b, 'batched_beta_choice', {'beta': 0.5})
            
            # Store results
            results_data.extend([
                {'batch_size': b, 'n': n, 'lambda': lam, 'Strategy': 'Batched One-choice', 
                 'Max_Load': one_result['avg_max_load'], 'Gap': one_result['avg_gap'],
                 'Std_Dev': one_result['avg_std_dev']},
                {'batch_size': b, 'n': n, 'lambda': lam, 'Strategy': 'Batched Two-choice', 
                 'Max_Load': two_result['avg_max_load'], 'Gap': two_result['avg_gap'],
                 'Std_Dev': two_result['avg_std_dev']},
                {'batch_size': b, 'n': n, 'lambda': lam, 'Strategy': 'Batched (1+β)-choice', 
                 'Max_Load': beta_result['avg_max_load'], 'Gap': beta_result['avg_gap'],
                 'Std_Dev': beta_result['avg_std_dev']}
            ])
    
    # Create DataFrame and generate plots
    df_results = pd.DataFrame(results_data)
    print("\nAll batched experiments completed!")
    
    # Generate comparison plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Gap vs lambda for different batch sizes (Two-choice)
    plt.subplot(2, 2, 1)
    for b in b_values[:4]:  # Plot first 4 batch sizes for clarity
        b_data = df_results[df_results['batch_size'] == b]
        two_choice_data = b_data[b_data['Strategy'] == 'Batched Two-choice']
        
        if not two_choice_data.empty:
            plt.plot(two_choice_data['lambda'], two_choice_data['Gap'], 
                    marker='o', label=f'b={b}', linewidth=2, markersize=6)
    
    plt.xlabel('λ = m²/b')
    plt.ylabel('Gap (Gₙ)')
    plt.title('Batched Two-choice: Gap vs λ for Different Batch Sizes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Comparison of strategies for fixed batch size
    plt.subplot(2, 2, 2)
    fixed_b = 1000  # Example batch size
    b_data = df_results[df_results['batch_size'] == fixed_b]
    
    strategies = ['Batched One-choice', 'Batched Two-choice', 'Batched (1+β)-choice']
    colors = ['red', 'blue', 'green']
    
    for i, strategy in enumerate(strategies):
        strategy_data = b_data[b_data['Strategy'] == strategy]
        if not strategy_data.empty:
            plt.plot(strategy_data['lambda'], strategy_data['Gap'], 
                    marker='o', color=colors[i], label=strategy, linewidth=2, markersize=6)
    
    plt.xlabel('λ = m²/b')
    plt.ylabel('Gap (Gₙ)')
    plt.title(f'Strategy Comparison (b={fixed_b})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Maximum load vs lambda
    plt.subplot(2, 2, 3)
    for b in b_values[:3]:
        b_data = df_results[df_results['batch_size'] == b]
        two_choice_data = b_data[b_data['Strategy'] == 'Batched Two-choice']
        
        if not two_choice_data.empty:
            plt.plot(two_choice_data['lambda'], two_choice_data['Max_Load'], 
                    marker='s', label=f'b={b}', linewidth=2, markersize=6)
    
    plt.xlabel('λ = m²/b')
    plt.ylabel('Maximum Load')
    plt.title('Maximum Load vs λ for Different Batch Sizes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Standard deviation vs lambda
    plt.subplot(2, 2, 4)
    for b in b_values[:3]:
        b_data = df_results[df_results['batch_size'] == b]
        two_choice_data = b_data[b_data['Strategy'] == 'Batched Two-choice']
        
        if not two_choice_data.empty:
            plt.plot(two_choice_data['lambda'], two_choice_data['Std_Dev'], 
                    marker='^', label=f'b={b}', linewidth=2, markersize=6)
    
    plt.xlabel('λ = m²/b')
    plt.ylabel('Standard Deviation')
    plt.title('Load Distribution Std Dev vs λ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('batched_allocations_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Batched allocation plots generated!")
    
    # Display summary table
    print("\n" + "=" * 70)
    print("SUMMARY RESULTS TABLE")
    print("=" * 70)
    
    # Create pivot table for better organization
    pivot_df = df_results.pivot_table(
        index=['batch_size', 'lambda'], 
        columns='Strategy', 
        values=['Gap', 'Max_Load', 'Std_Dev']
    )
    
    # Display formatted table
    with pd.option_context('display.float_format', '{:.2f}'.format):
        print(pivot_df.round(2))
    
    print("PROGRAM COMPLETED")