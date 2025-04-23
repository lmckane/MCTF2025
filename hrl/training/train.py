import numpy as np
import torch
import argparse
from typing import Dict, Any
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
from hrl.utils.opponent_modeler import OpponentModeler

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
            'difficulty': 0.2,
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
            'options': ['attack_flag', 'capture_flag', 'guard_flag', 'evade', 'retreat'],
            'debug_level': args.debug_level
        },
        'training_config': {
            'num_episodes': args.num_episodes,
            'batch_size': 32,
            'log_interval': args.log_interval,
            'checkpoint_interval': args.checkpoint_interval,
            'render': args.render,
            'log_dir': args.log_dir,
            'debug_level': args.debug_level,
            'metrics_config': {
                'log_dir': args.log_dir,
                'log_interval': args.log_interval,
                'save_replays': args.save_replays
            }
        },
        'curriculum_config': {
            'stages': [
                {'name': 'basic', 'difficulty': 0.2, 'duration': 0.2},
                {'name': 'intermediate', 'difficulty': 0.5, 'duration': 0.3},
                {'name': 'advanced', 'difficulty': 0.8, 'duration': 0.5}
            ]
        },
        'adversarial_config': {
            'ratio': 0.3,
            'update_freq': 100
        }
    }
    return config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train an agent for the Capture The Flag game.')
    
    # Training parameters
    parser.add_argument('--num-episodes', type=int, default=5000,
                       help='Number of episodes to train for')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode')
    parser.add_argument('--num-agents', type=int, default=3,
                       help='Number of agents per team')
    
    # Logging parameters
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory to save logs')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Episodes between logging')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                       help='Episodes between saving checkpoints')
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
    parser.add_argument('--save-model', type=str, default='final_model',
                       help='Name for final model (without extension)')
    parser.add_argument('--checkpoint-dir', type=str, default=os.path.join('hrl', 'checkpoints'),
                       help='Directory to save checkpoints')
    
    return parser.parse_args()

def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    
    # Create log directory if needed
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create checkpoint directory if needed
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Starting training with the following parameters:")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  Agents per team: {args.num_agents}")
    print(f"  Max steps per episode: {args.max_steps}")
    print(f"  Debug level: {args.debug_level}")
    print(f"  Log directory: {args.log_dir}")
    print(f"  Checkpoint directory: {args.checkpoint_dir}")
    print(f"{'='*80}\n")
    
    # Create config
    config = create_config(args)
    
    # Create environment
    env = GameEnvironment(config['env_config'])
    
    # Create state processor
    state_processor = StateProcessor(config['policy_config'])
    
    # Create option selector
    option_selector = OptionSelector(config['policy_config'])
    
    # Create team coordinator
    team_coordinator = TeamCoordinator(config['curriculum_config']['stages'][0])
    
    # Create opponent modeler
    opponent_modeler = OpponentModeler(config['curriculum_config']['stages'][0])
    
    # Create policy
    policy = HierarchicalPolicy(
        state_size=state_processor.state_size,
        action_size=2,  # 2D movement
        config=config['policy_config'],
        option_selector=option_selector,
        state_processor=state_processor
    )
    
    # Load model if specified
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        success = trainer.load_checkpoint(args.load_model)
        if not success:
            print("Failed to load model, starting with a new model.")
    
    # Start timer
    start_time = time.time()
    
    # Create metrics tracking
    metrics = Metrics()
    
    # Create trainer
    print("Initializing trainer...")
    trainer = Trainer(env, policy, metrics, config)
    trainer.team_coordinator = team_coordinator
    trainer.opponent_modeler = opponent_modeler
    
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
    
    # Save final model
    trainer.save_checkpoint(args.save_model)
    
    # Determine the actual path of the saved model
    model_path = os.path.join(args.checkpoint_dir, f"{args.save_model}.pth")
    print(f"Model saved to {model_path}")
    
    # Visualization reminder
    print("\nTo visualize training metrics, run:")
    print(f"python hrl/visualization/plot_metrics.py --log-dir {args.log_dir} --output-dir plots")

if __name__ == '__main__':
    main() 