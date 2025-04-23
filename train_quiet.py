#!/usr/bin/env python
"""
Helper script to run training with minimal output.
This script simply calls the main training script with debug level 0.
"""

import subprocess
import sys
import os
import argparse
import time

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run training with minimal output.')
    
    parser.add_argument('--episodes', type=int, default=5000,
                       help='Number of episodes to train for')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory to save logs')
    parser.add_argument('--output', type=str, default=None,
                       help='File to save output to (default: None)')
    parser.add_argument('--checkpoint-dir', type=str, default=os.path.join('hrl', 'checkpoints'),
                       help='Directory to save checkpoints')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Model to load for continued training')
    
    return parser.parse_args()

def main():
    """Run training with minimal output."""
    args = parse_args()
    
    # Ensure directories exist
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Construct command
    cmd = [
        'python', 'hrl/training/train.py',
        '--num-episodes', str(args.episodes),
        '--debug-level', '0',
        '--log-dir', args.log_dir,
        '--checkpoint-dir', args.checkpoint_dir,
        '--log-interval', '50',
        '--checkpoint-interval', '100'
    ]
    
    # Add load model if specified
    if args.load_model:
        cmd.extend(['--load-model', args.load_model])
    
    print(f"\n{'='*80}")
    print(f"Running training with {args.episodes} episodes")
    print(f"Logs will be saved to {args.log_dir}")
    print(f"Checkpoints will be saved to {args.checkpoint_dir}")
    if args.load_model:
        print(f"Loading model from {args.load_model}")
    print(f"{'='*80}\n")
    
    # Timestamp for the run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    if args.output:
        output_file = args.output
        print(f"Output will be saved to {output_file}")
        with open(output_file, 'w') as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=f)
    else:
        # If no output file specified, create one with timestamp
        output_file = os.path.join(args.log_dir, f"training-{timestamp}.log")
        print(f"Output will be saved to {output_file}")
        with open(output_file, 'w') as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=f)
    
    print("\nTraining is running in the background.")
    print("You can monitor progress with:")
    print(f"  tail -f {output_file}")
    print("\nTo visualize metrics after training, run:")
    print(f"python hrl/visualization/plot_metrics.py --log-dir {args.log_dir} --output-dir plots")

if __name__ == '__main__':
    main() 