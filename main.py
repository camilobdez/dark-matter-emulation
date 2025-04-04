import os
import subprocess
import pickle
import matplotlib.pyplot as plt
import numpy as np

def run_all_sizes(train_sizes, results_dir="results"):
    """
    Run the model for all specified training sizes, either by running the script
    or loading existing results.
    
    Parameters:
    -----------
    train_sizes : list
        List of training sizes to evaluate
    results_dir : str
        Directory to save/load results
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Check which sizes have already been processed
    existing_results = [f for f in os.listdir(results_dir) if f.startswith("result_n") and f.endswith(".pkl")]
    existing_sizes = [int(f.split("_n")[1].split(".pkl")[0]) for f in existing_results]
    
    # Process each training size
    for size in train_sizes:
        print("SIZE: ", size)
        if size not in existing_sizes:
            print(f"Running model for training size {size}...")
            subprocess.run(["python", "train_model.py", "--train_size", str(size), "--save_dir", results_dir])
        else:
            print(f"Results for training size {size} already exist. Skipping.")

def plot_results(train_sizes, results_dir="results"):
    """
    Load results and create plot
    
    Parameters:
    -----------
    train_sizes : list
        List of training sizes to include in plot
    results_dir : str
        Directory containing results
    """
    # Load results
    avg_log_scores = []
    sizes_with_results = []
    
    for size in train_sizes:
        result_file = f"{results_dir}/result_n{size}.pkl"
        if os.path.exists(result_file):
            with open(result_file, 'rb') as f:
                result = pickle.load(f)
                avg_log_scores.append(result['avg_log_score'])
                sizes_with_results.append(size)
        else:
            print(f"Warning: No results found for training size {size}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes_with_results, avg_log_scores, 'o-', markersize=8, linewidth=2, color='blue')
    plt.xlabel('Number of Training Samples', fontsize=12)
    plt.ylabel('Average Log Score per Test Image', fontsize=12)
    plt.title('Test Log Score vs. Training Sample Size', fontsize=14)
    plt.grid(True)
    plt.xscale('log')  # Using log scale for x-axis
    plt.xticks(sizes_with_results, [str(n) for n in sizes_with_results])
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'{results_dir}/log_score_vs_training_size.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # Define training sample sizes
    train_sizes = [5, 10, 30, 50, 100, 160]
    
    # Directory to store results
    results_dir = "log_score_results"
    
    # Run models for all sizes if needed
    run_all_sizes(train_sizes, results_dir)
    
    # Create plot
    plot_results(train_sizes, results_dir)