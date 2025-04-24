import numpy as np
import torch
import argparse
from typing import Dict, Any, List
import time
import os
from hrl.training.trainer import Trainer
from hrl.utils.metrics import MetricsTracker
from hrl.policies.hierarchical_policy import HierarchicalPolicy
from hrl.utils.option_selector import OptionSelector
from hrl.utils.state_processor import StateProcessor
from hrl.environment.game_env import GameEnvironment
from hrl.utils.metrics import Metrics
from hrl.utils.team_coordinator import TeamCoordinator
from hrl.utils.opponent_modeler import OpponentModeler, OpponentStrategy
from datetime import datetime

def create_config(args) -> Dict[str, Any]:
    """Create configuration for training."""
    config = {
        'env_config': {
            'map_size': [100, 100],
            'num_agents': args.num_agents,
            'max_steps': args.max_steps,
            'tag_radius': 5,
            'capture_radius': 10,
            'base_radius': 20,
            'difficulty': 0.2,  # Initial difficulty, will be adjusted by curriculum
            'max_velocity': 5.0,
            'debug_level': args.debug_level
        },
        'policy_config': {
            'state_size': 8,
            'action_size': 2,
            'hidden_size': 64,
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'lambda_': 0.95,
            'entropy_coef': 0.01,
            'options': ['attack_flag', 'capture_flag', 'guard_flag', 'evade', 'retreat', 'return_to_base'],
            'debug_level': args.debug_level,
            'buffer_size': 10000,  # Increased replay buffer size
            'advantage_norm': True,  # Normalize advantages for more stable training
        },
        'training_config': {
            'num_episodes': args.num_episodes,
            'batch_size': 64,  # Increased batch size
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
            'max_checkpoints': args.max_checkpoints,  # Number of previous model versions to keep
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
                    'difficulty': 0.2,
                    'opponent_strategies': {
                        'random': 0.7,
                        'direct': 0.3,
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
                        'direct': 0.4,
                        'defensive': 0.2,
                        'coordinated': 0.1
                    },
                    'duration': 0.3  # 30% of total episodes
                },
                # Stage 3: Advanced opponents with significant coordination
                {
                    'name': 'advanced',
                    'difficulty': 0.8,
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
                    'difficulty': 1.0,
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
            'num_opponents': args.num_agents,
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

def list_available_models(checkpoint_dir):
    """List all available models in the checkpoint directory."""
    import glob
    import os
    from datetime import datetime
    
    # Ensure checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return
    
    # Find all model files
    model_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    versioned_files = glob.glob(os.path.join(checkpoint_dir, "*_v*.pth"))
    
    # Only include non-versioned files in the main list
    main_models = [f for f in model_files if f not in versioned_files]
    
    if not main_models and not versioned_files:
        print("No models found.")
        return
    
    # Group versioned files by base name
    version_groups = {}
    for vfile in versioned_files:
        basename = os.path.basename(vfile)
        # Extract base name before the _v timestamp
        if "_v" in basename:
            base = basename.split("_v")[0]
            if base not in version_groups:
                version_groups[base] = []
            version_groups[base].append(vfile)
    
    # Print main models
    print(f"\n{'='*80}")
    print(f"AVAILABLE MODELS in {checkpoint_dir}:")
    print(f"{'-'*80}")
    
    if main_models:
        print("Main Models:")
        for model in sorted(main_models):
            basename = os.path.basename(model)
            timestamp = datetime.fromtimestamp(os.path.getmtime(model)).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  - {basename} (Last modified: {timestamp})")
    
    # Print versioned models
    if version_groups:
        print("\nVersioned Models:")
        for base, versions in sorted(version_groups.items()):
            print(f"  {base} ({len(versions)} versions):")
            # Sort by creation time, newest first
            for v in sorted(versions, key=os.path.getctime, reverse=True)[:3]:  # Show only most recent 3
                basename = os.path.basename(v)
                timestamp = datetime.fromtimestamp(os.path.getmtime(v)).strftime('%Y-%m-%d %H:%M:%S')
                print(f"    - {basename} (Created: {timestamp})")
            if len(versions) > 3:
                print(f"    - ... and {len(versions) - 3} more versions")
    
    print(f"{'='*80}")
    
    return main_models, version_groups

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train an agent for the Capture The Flag game.')
    
    # Get current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Training parameters
    parser.add_argument('--num-episodes', type=int, default=10000,
                       help='Number of episodes to train for (default: 10000)')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode (default: 500)')
    parser.add_argument('--num-agents', type=int, default=3,
                       help='Number of agents per team (default: 3)')
    
    # Curriculum and training settings
    parser.add_argument('--no-curriculum', action='store_true',
                       help='Disable curriculum learning')
    parser.add_argument('--no-self-play', action='store_true',
                       help='Disable self-play learning')
    parser.add_argument('--opponent-strategy', type=str, default=None, choices=['random', 'direct', 'defensive', 'aggressive', 'coordinated'],
                       help='Fixed opponent strategy (overrides curriculum)')
    
    # Logging parameters
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory to save logs (default: logs)')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Episodes between logging (default: 10)')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                       help='Episodes between saving checkpoints (default: 100)')
    parser.add_argument('--debug-level', type=int, default=1,
                       help='Debug level (0=minimal, 1=normal, 2=verbose)')
    
    # Visualization
    parser.add_argument('--render', action='store_true',
                       help='Render the environment during training')
    parser.add_argument('--save-replays', action='store_true',
                       help='Save game replays during training')
    
    # Model loading/saving
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to model to load')
    parser.add_argument('--model-prefix', type=str, default='model',
                       help='Prefix for the model name (default: "model")')
    parser.add_argument('--save-model', type=str, default=None,
                       help='Name for final model (default: "<prefix>_<timestamp>")')
    parser.add_argument('--checkpoint-dir', type=str, default=os.path.join('hrl', 'checkpoints'),
                       help='Directory to save checkpoints')
    parser.add_argument('--max-checkpoints', type=int, default=5,
                       help='Number of previous model versions to keep (default: 5)')
    
    # Utility options
    parser.add_argument('--list-models', action='store_true',
                       help='List all available models in checkpoint directory and exit')
    
    args = parser.parse_args()
    
    # Generate default model name with timestamp if not specified
    if args.save_model is None:
        args.save_model = f"{args.model_prefix}_{timestamp}"
    
    return args

def adjust_config_from_args(config, args):
    """Adjust configuration based on command line arguments."""
    # Handle fixed opponent strategy if specified
    if args.opponent_strategy:
        for stage in config['curriculum_config']['stages']:
            # Set all probabilities to 0 except the chosen strategy
            for strategy in stage['opponent_strategies']:
                stage['opponent_strategies'][strategy] = 0.0
            # Set the chosen strategy to 1.0 if available, otherwise keep original
            if args.opponent_strategy in stage['opponent_strategies']:
                stage['opponent_strategies'][args.opponent_strategy] = 1.0
    
    # Disable curriculum learning if requested
    if args.no_curriculum:
        config['curriculum_config']['enabled'] = False
    
    # Disable self-play if requested
    if args.no_self_play:
        config['self_play_config']['enabled'] = False
    
    return config

def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    
    # Handle utility functions
    if args.list_models:
        list_available_models(args.checkpoint_dir)
        return
    
    # Create log directory if needed
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create checkpoint directory if needed
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Starting training with the following parameters:")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Agents per team: {args.num_agents}")
    print(f"  Max steps per episode: {args.max_steps}")
    print(f"  Curriculum learning: {'Enabled' if not args.no_curriculum else 'Disabled'}")
    print(f"  Self-play: {'Enabled' if not args.no_self_play else 'Disabled'}")
    if args.opponent_strategy:
        print(f"  Fixed opponent strategy: {args.opponent_strategy}")
    print(f"  Debug level: {args.debug_level}")
    print(f"  Log directory: {args.log_dir}")
    print(f"  Checkpoint directory: {args.checkpoint_dir}")
    print(f"  Checkpoint interval: Every {args.checkpoint_interval} episodes")
    print(f"  Preserving previous models: {args.max_checkpoints} versions")
    print(f"  Model prefix: {args.model_prefix}")
    print(f"  Final model will be saved as: {args.save_model}")
    print(f"{'='*80}\n")
    
    # Create config
    config = create_config(args)
    config = adjust_config_from_args(config, args)
    
    # Start timer
    start_time = time.time()
    
    # Create trainer
    print("Initializing trainer...")
    trainer = Trainer(config)
    
    # Load model if specified
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        success = trainer.load_checkpoint(args.load_model)
        if not success:
            print("Failed to load model, starting with a new model.")
    
    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        
    # End timer
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Save model with timestamped name
    trainer.save_checkpoint(args.save_model)
    
    # Also save as final_model for backward compatibility
    if args.save_model != "final_model":
        print("Also saving a copy as 'final_model' for backward compatibility")
        trainer.save_checkpoint("final_model", preserve_history=False)
    
    # Determine the actual path of the saved model
    model_path = os.path.join(args.checkpoint_dir, f"{args.save_model}.pth")
    print(f"Model saved to {model_path}")
    
    # Visualization reminder
    print("\nTo visualize training metrics, run:")
    print(f"python hrl/visualization/plot_metrics.py --log-dir {args.log_dir} --output-dir plots")

if __name__ == '__main__':
    main() 