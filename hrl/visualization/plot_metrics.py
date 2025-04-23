import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from glob import glob
import sys

def load_metrics(metrics_file):
    """Load metrics from JSON file."""
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metrics from {metrics_file}: {e}")
        return None

def smooth_data(data, window=10):
    """Apply smoothing to data series."""
    if len(data) < window:
        return data
    weights = np.ones(window) / window
    return np.convolve(data, weights, mode='valid')

def plot_metrics(metrics_files, output_dir=None, window_size=10):
    """Plot metrics from files."""
    if not metrics_files:
        print("No metrics files found!")
        return
    
    # Prepare data
    all_metrics = []
    for file in metrics_files:
        metrics = load_metrics(file)
        if metrics:
            all_metrics.append(metrics)
    
    if not all_metrics:
        print("No valid metrics data found!")
        return
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Collect all metrics keys
    metric_keys = set()
    for metrics in all_metrics:
        metric_keys.update(metrics.keys())
    
    # Filter out non-list metrics and special keys
    plottable_metrics = [key for key in metric_keys 
                        if isinstance(all_metrics[0].get(key, []), list) 
                        and key not in ['timestamp']]
    
    # Group metrics by type
    metric_groups = {
        'rewards': [k for k in plottable_metrics if 'reward' in k.lower()],
        'win_rate': [k for k in plottable_metrics if 'win' in k.lower()],
        'scores': [k for k in plottable_metrics if 'score' in k.lower()],
        'episodes': [k for k in plottable_metrics if 'episode' in k.lower()],
        'options': [k for k in plottable_metrics if 'option' in k.lower()],
        'other': []
    }
    
    # Add other metrics
    for metric in plottable_metrics:
        if not any(metric in group for group in metric_groups.values()):
            metric_groups['other'].append(metric)
    
    # Create plots for each group
    for group_name, metrics in metric_groups.items():
        if not metrics:
            continue
        
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle(f"{group_name.capitalize()} Metrics", fontsize=16)
        
        for metric in metrics:
            for i, metrics_data in enumerate(all_metrics):
                if metric not in metrics_data:
                    continue
                    
                data = metrics_data[metric]
                if not data or not isinstance(data, list):
                    continue
                
                # Convert all elements to float
                try:
                    data = [float(x) if x is not None else 0.0 for x in data]
                except (ValueError, TypeError):
                    print(f"Skipping non-numeric data in {metric}")
                    continue
                
                # Apply smoothing
                if len(data) >= window_size:
                    smoothed_data = smooth_data(data, window_size)
                    episodes = range(len(smoothed_data))
                    ax.plot(episodes, smoothed_data, label=f"{metric} (file {i+1})")
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{group_name}_metrics.png"))
            print(f"Saved {group_name} plot to {os.path.join(output_dir, f'{group_name}_metrics.png')}")
        else:
            plt.show()
        
        plt.close()
    
    # Create a combined plot for the most important metrics
    create_combined_plot(all_metrics, window_size, output_dir)
    
def create_combined_plot(all_metrics, window_size, output_dir=None):
    """Create a combined plot with the most important metrics."""
    key_metrics = ['win_rate', 'avg_reward', 'avg_score']
    
    fig, axes = plt.subplots(len(key_metrics), 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Training Progress', fontsize=16)
    
    for i, metric in enumerate(key_metrics):
        for j, metrics_data in enumerate(all_metrics):
            # Find the closest matching key if exact key not found
            matching_key = None
            for key in metrics_data:
                if metric.lower() in key.lower() and isinstance(metrics_data[key], list):
                    matching_key = key
                    break
            
            if not matching_key:
                continue
                
            data = metrics_data[matching_key]
            if not data:
                continue
            
            # Convert all elements to float
            try:
                data = [float(x) if x is not None else 0.0 for x in data]
            except (ValueError, TypeError):
                continue
            
            # Apply smoothing
            if len(data) >= window_size:
                smoothed_data = smooth_data(data, window_size)
                episodes = range(len(smoothed_data))
                axes[i].plot(episodes, smoothed_data, label=f"File {j+1}")
        
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # Add horizontal line at win rate 0.5
        if 'win' in metric.lower():
            axes[i].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    
    # Add common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    
    # Set common x-label
    axes[-1].set_xlabel('Episode')
    
    plt.tight_layout()
    
    # Save plot
    if output_dir:
        plt.savefig(os.path.join(output_dir, "training_progress.png"))
        print(f"Saved combined plot to {os.path.join(output_dir, 'training_progress.png')}")
    else:
        plt.show()
    
    plt.close()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Plot training metrics.')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory containing metrics files')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Directory to save plots')
    parser.add_argument('--pattern', type=str, default='metrics_*.json',
                       help='File pattern to match metrics files')
    parser.add_argument('--window', type=int, default=10,
                       help='Smoothing window size')
    args = parser.parse_args()
    
    # Find metrics files
    metrics_files = glob(os.path.join(args.log_dir, args.pattern))
    
    if not metrics_files:
        print(f"No metrics files found in {args.log_dir} matching pattern {args.pattern}")
        return
    
    print(f"Found {len(metrics_files)} metrics files")
    
    # Plot metrics
    plot_metrics(metrics_files, args.output_dir, args.window)

if __name__ == '__main__':
    main() 