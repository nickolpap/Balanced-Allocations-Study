import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

def get_median_threshold(bins):
    """Calculate median load threshold"""
    return np.median(bins)

def get_percentile_threshold(bins, percentile):
    """Calculate percentile threshold (25% or 75%)"""
    return np.percentile(bins, percentile)

def is_above_median(bins, bin_index):
    """Check if bin load is above median"""
    median = get_median_threshold(bins)
    return bins[bin_index] > median

def is_in_top_percentile(bins, bin_index, percentile):
    """Check if bin is in top X% loaded bins"""
    threshold = get_percentile_threshold(bins, percentile)
    return bins[bin_index] >= threshold

def partial_info_one_question(n, m):
    """
    Partial information with k=1 question:
    "Is the bin load above median?"
    """
    bins = [0] * m
    
    for _ in range(n):
        # Choose two candidate bins
        bin1 = random.randint(0, m-1)
        bin2 = random.randint(0, m-1)
        
        # Ask one question for each candidate
        above_median1 = is_above_median(bins, bin1)
        above_median2 = is_above_median(bins, bin2)
        
        # Decision logic
        if above_median1 and not above_median2:
            # bin2 is below median, choose bin2
            bins[bin2] += 1
        elif above_median2 and not above_median1:
            # bin1 is below median, choose bin1
            bins[bin1] += 1
        else:
            # Both above or both below median - choose randomly
            if random.random() < 0.5:
                bins[bin1] += 1
            else:
                bins[bin2] += 1
                
    return bins

def partial_info_two_questions(n, m):
    """
    Partial information with k=2 questions:
    1st: "Above median?"
    2nd: "In top 25% or 75%?" based on first answer
    """
    bins = [0] * m
    
    for _ in range(n):
        bin1 = random.randint(0, m-1)
        bin2 = random.randint(0, m-1)
        
        # First question for both bins
        above_median1 = is_above_median(bins, bin1)
        above_median2 = is_above_median(bins, bin2)
        
        # Case 1: One above median, one below
        if above_median1 and not above_median2:
            bins[bin2] += 1
        elif above_median2 and not above_median1:
            bins[bin1] += 1
        
        # Case 2: Both below median
        elif not above_median1 and not above_median2:
            # Second question: Are they in top 75%?
            in_top_75_1 = is_in_top_percentile(bins, bin1, 75)
            in_top_75_2 = is_in_top_percentile(bins, bin2, 75)
            
            if in_top_75_1 and not in_top_75_2:
                bins[bin2] += 1
            elif in_top_75_2 and not in_top_75_1:
                bins[bin1] += 1
            else:
                # Tie - choose randomly
                if random.random() < 0.5:
                    bins[bin1] += 1
                else:
                    bins[bin2] += 1
        
        # Case 3: Both above median
        else:
            # Second question: Are they in top 25%?
            in_top_25_1 = is_in_top_percentile(bins, bin1, 25)
            in_top_25_2 = is_in_top_percentile(bins, bin2, 25)
            
            if in_top_25_1 and not in_top_25_2:
                bins[bin2] += 1
            elif in_top_25_2 and not in_top_25_1:
                bins[bin1] += 1
            else:
                # Tie - choose randomly
                if random.random() < 0.5:
                    bins[bin1] += 1
                else:
                    bins[bin2] += 1
                    
    return bins

def run_partial_info_experiment(n, m, strategy, num_runs=15):
    """
    Run partial information experiment multiple times
    """
    max_loads = []
    gaps = []
    std_devs = []
    
    for _ in range(num_runs):
        if strategy == 'one_question':
            bins = partial_info_one_question(n, m)
        elif strategy == 'two_questions':
            bins = partial_info_two_questions(n, m)
        
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

# Main execution for partial information experiments
if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING PARTIAL INFORMATION ALLOCATION EXPERIMENTS")
    print("=" * 80)
    
    m = 100
    n_values = [100, 500, 1000, 2500, 5000, 10000]
    
    results_data = []
    
    for n in n_values:
        print(f"\n--- n = {n} (n/m = {n/m}) ---")
        
        # Run partial information strategies
        one_question_result = run_partial_info_experiment(n, m, 'one_question')
        two_questions_result = run_partial_info_experiment(n, m, 'two_questions')
        
        print(f"1-question:    Max Load = {one_question_result['avg_max_load']:6.2f} ± {one_question_result['std_max_load']:4.2f}, "
              f"Gap = {one_question_result['avg_gap']:6.2f}")
        print(f"2-questions:   Max Load = {two_questions_result['avg_max_load']:6.2f} ± {two_questions_result['std_max_load']:4.2f}, "
              f"Gap = {two_questions_result['avg_gap']:6.2f}")
        
        # Store results
        results_data.extend([
            {'n': n, 'Strategy': '1-question', 
             'Max_Load': one_question_result['avg_max_load'], 
             'Gap': one_question_result['avg_gap'],
             'Std_Dev': one_question_result['avg_std_dev']},
            {'n': n, 'Strategy': '2-questions', 
             'Max_Load': two_questions_result['avg_max_load'], 
             'Gap': two_questions_result['avg_gap'],
             'Std_Dev': two_questions_result['avg_std_dev']}
        ])
    
    # Create DataFrame and generate plots
    df_results = pd.DataFrame(results_data)
    print("\nAll partial information experiments completed!")
    
    # Generate comparison plots
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Gap vs n for both strategies
    plt.subplot(2, 2, 1)
    strategies = ['1-question', '2-questions']
    colors = ['blue', 'red']
    
    for i, strategy in enumerate(strategies):
        strategy_data = df_results[df_results['Strategy'] == strategy]
        plt.plot(strategy_data['n'], strategy_data['Gap'], 
                 marker='o', color=colors[i], linewidth=2, 
                 label=strategy, markersize=6)
    
    plt.xlabel('Number of Balls (n)')
    plt.ylabel('Gap (Gₙ)')
    plt.title('Partial Information: Gap vs Number of Balls')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Maximum Load vs n
    plt.subplot(2, 2, 2)
    for i, strategy in enumerate(strategies):
        strategy_data = df_results[df_results['Strategy'] == strategy]
        plt.plot(strategy_data['n'], strategy_data['Max_Load'], 
                 marker='s', color=colors[i], linewidth=2, 
                 label=strategy, markersize=6)
    
    plt.xlabel('Number of Balls (n)')
    plt.ylabel('Maximum Load')
    plt.title('Partial Information: Maximum Load vs n')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Standard Deviation vs n
    plt.subplot(2, 2, 3)
    for i, strategy in enumerate(strategies):
        strategy_data = df_results[df_results['Strategy'] == strategy]
        plt.plot(strategy_data['n'], strategy_data['Std_Dev'], 
                 marker='^', color=colors[i], linewidth=2, 
                 label=strategy, markersize=6)
    
    plt.xlabel('Number of Balls (n)')
    plt.ylabel('Standard Deviation')
    plt.title('Partial Information: Std Dev vs n')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Improvement from additional question
    plt.subplot(2, 2, 4)
    one_question_data = df_results[df_results['Strategy'] == '1-question']
    two_questions_data = df_results[df_results['Strategy'] == '2-questions']
    improvement_ratio = one_question_data['Gap'].values / two_questions_data['Gap'].values
    
    plt.plot(one_question_data['n'], improvement_ratio, 'green', marker='D', 
             linewidth=2, markersize=6, label='Improvement Ratio')
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('Number of Balls (n)')
    plt.ylabel('Improvement Ratio (1Q/2Q)')
    plt.title('Benefit of Second Question')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('partial_info_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Partial information plots generated!")
    
    # Display summary table
    print("\n" + "=" * 70)
    print("SUMMARY RESULTS TABLE")
    print("=" * 70)
    
    # Create pivot table
    pivot_df = df_results.pivot_table(
        index='n', 
        columns='Strategy', 
        values=['Max_Load', 'Gap', 'Std_Dev']
    )
    
    # Display formatted table
    with pd.option_context('display.float_format', '{:.2f}'.format):
        print(pivot_df.round(2))
    
    print("PROGRAM COMPLETED")