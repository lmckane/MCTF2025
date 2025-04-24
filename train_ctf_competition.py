"""
Training script for 3v3 Capture The Flag competition with improved team coordination.
Uses the official Pyquaticus environment for training.
"""

import numpy as np
import torch
import argparse
import os
from datetime import datetime
import time
from tqdm import tqdm
import sys

# Add the project root to Python path
sys.path.append('.')

from hrl.training.trainer import Trainer
from hrl.utils.metrics import MetricsTracker
from hrl.policies.hierarchical_policy import HierarchicalPolicy
from hrl.utils.option_selector import OptionSelector
from hrl.utils.state_processor import StateProcessor
from hrl.environment.pyquaticus_wrapper import PyquaticusWrapper
from hrl.utils.metrics import Metrics
from hrl.utils.team_coordinator import TeamCoordinator
from hrl.utils.opponent_modeler import OpponentModeler, OpponentStrategy

def create_config(args):
    """Create configuration for training."""
    config = {
        'env_config': {
            'map_size': [100, 100],
            'num_agents': 3,  # Fixed to 3v3 for the competition
            'max_steps': 1000,
            'tag_radius': 5,
            'capture_radius': 10,
            'base_radius': 20,
            'difficulty': 0.5,  # Start with medium difficulty
            'max_velocity': 5.0,
            'debug_level': args.debug_level
        },
        'policy_config': {
            'state_size': 8,
            'action_size': 2,
            'hidden_size': 128,  # Larger network capacity
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'lambda_': 0.95,
            'entropy_coef': 0.01,
            'options': ['attack_flag', 'capture_flag', 'guard_flag', 'evade', 'retreat', 'return_to_base'],
            'debug_level': args.debug_level,
            'buffer_size': 20000,  # Increased replay buffer size
            'advantage_norm': True,  # Normalize advantages for more stable training
        },
        'training_config': {
            'num_episodes': args.num_episodes,
            'batch_size': 128,  # Larger batch size for more stable updates
            'log_interval': args.log_interval,
            'checkpoint_interval': args.checkpoint_interval,
            'render': args.render,
            'log_dir': args.log_dir,
            'checkpoint_dir': args.checkpoint_dir,
            'debug_level': args.debug_level,
            'metrics_config': {
                'log_dir': args.log_dir,
                'log_interval': args.log_interval,
                'save_replays': args.save_replays
            },
            'eval_interval': 100,  # Evaluate policy every 100 episodes
            'early_stopping_patience': 20,  # Stop if no improvement for 20 evaluations
            'max_checkpoints': args.max_checkpoints,
        },
        'curriculum_config': {
            'enabled': True,
            'progression_metric': 'win_rate',  # Progress based on win rate
            'progression_threshold': 0.6,  # Progress when win rate > 60%
            'min_episodes_per_stage': 500,  # Minimum episodes before stage transition
            'stages': [
                # Stage 1: Basic opponents with minimal coordination
                {
                    'name': 'basic',
                    'difficulty': 0.3,
                    'opponent_strategies': {
                        'random': 0.5,
                        'direct': 0.5,
                        'coordinated': 0.0
                    },
                    'duration': 0.2  # 20% of total episodes
                },
                # Stage 2: Intermediate opponents with some coordination
                {
                    'name': 'intermediate',
                    'difficulty': 0.5,
                    'opponent_strategies': {
                        'random': 0.3,
                        'direct': 0.3,
                        'defensive': 0.2,
                        'coordinated': 0.2
                    },
                    'duration': 0.3  # 30% of total episodes
                },
                # Stage 3: Advanced opponents with significant coordination
                {
                    'name': 'advanced',
                    'difficulty': 0.7,
                    'opponent_strategies': {
                        'random': 0.1, 
                        'direct': 0.2,
                        'defensive': 0.3,
                        'aggressive': 0.2,
                        'coordinated': 0.2
                    },
                    'duration': 0.3  # 30% of total episodes
                },
                # Stage 4: Expert opponents with high coordination
                {
                    'name': 'expert',
                    'difficulty': 0.9,
                    'opponent_strategies': {
                        'defensive': 0.2,
                        'aggressive': 0.3,
                        'coordinated': 0.5
                    },
                    'duration': 0.2  # 20% of total episodes
                }
            ]
        },
        'self_play_config': {
            'enabled': True,
            'start_episode': 1000,  # Start self-play after 1000 episodes
            'frequency': 0.3,  # 30% of episodes use self-play after start_episode
            'policy_bank_size': 5,  # Keep 5 past versions of the policy
            'policy_bank_update_freq': 500,  # Add current policy to bank every 500 episodes
            'team_play': True,  # Enable team coordination in self-play
            'opponent_sampling': 'prioritized',  # 'uniform', 'recent', or 'prioritized'
        },
        'opponent_config': {
            'num_opponents': 3,  # Fixed to 3 opponents for 3v3
            'strategies': {
                'random': {
                    'description': 'Random movement with some bias toward flags',
                    'implementation': 'random_strategy'
                },
                'direct': {
                    'description': 'Direct path to objective',
                    'implementation': 'direct_strategy'
                },
                'defensive': {
                    'description': 'Focus on defending own flag',
                    'implementation': 'defensive_strategy'
                },
                'aggressive': {
                    'description': 'Focus on tagging opponents',
                    'implementation': 'aggressive_strategy'
                },
                'coordinated': {
                    'description': 'Team-based strategy with role assignments',
                    'implementation': 'coordinated_strategy',
                    'role_distribution': {
                        'attacker': 0.4,
                        'defender': 0.3,
                        'support': 0.3
                    }
                }
            }
        }
    }
    return config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train 3v3 CTF agents with improved team coordination.')
    
    parser.add_argument('--num-episodes', type=int, default=5000,
                       help='Number of episodes to train')
    parser.add_argument('--log-interval', type=int, default=100,
                       help='Interval for logging training statistics')
    parser.add_argument('--checkpoint-interval', type=int, default=1000,
                       help='Interval for saving model checkpoints')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment during training')
    parser.add_argument('--debug-level', type=int, default=1,
                       help='Debug level (0=minimal, 1=normal, 2=verbose)')
    parser.add_argument('--save-replays', action='store_true',
                       help='Save episode replays')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    parser.add_argument('--max-checkpoints', type=int, default=5,
                       help='Maximum number of checkpoints to keep')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory for logs')
    parser.add_argument('--checkpoint-dir', type=str, default=os.path.join('hrl', 'checkpoints', 'competition'),
                       help='Directory for checkpoints')
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    
    # Create log directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("Creating configuration...")
    # Create configuration
    config = create_config(args)
    
    print("Creating trainer...")
    # Create trainer
    try:
        trainer = Trainer(config)
        print("Trainer created successfully!")
    except Exception as e:
        print(f"ERROR creating trainer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Set up logging
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(args.log_dir, f"training-{current_time}.log")
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, args.resume)
        if os.path.exists(checkpoint_path):
            print(f"Resuming training from {checkpoint_path}")
            trainer.load_checkpoint(checkpoint_path)
        else:
            print(f"Checkpoint {checkpoint_path} not found, starting fresh training")
    
    # Train the model
    print(f"\n{'='*80}")
    print(f"Starting training with Pyquaticus environment for 3v3 competition")
    print(f"Training for {args.num_episodes} episodes")
    print(f"Logs will be saved to {log_file}")
    print(f"Checkpoints will be saved to {args.checkpoint_dir}")
    print(f"{'='*80}\n")
    
    print("Starting training...")
    try:
        trainer.train()
        print("Training completed successfully!")
    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save final model
    print("Saving final model...")
    final_model_path = os.path.join(args.checkpoint_dir, "competition_final_model.pth")
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == '__main__':
    main() 