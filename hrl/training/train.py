import numpy as np
import torch
from hrl.training.trainer import Trainer
from hrl.utils.metrics import MetricsTracker
from hrl.policies.hierarchical_policy import HierarchicalPolicy
from hrl.utils.option_selector import OptionSelector
from hrl.utils.state_processor import StateProcessor
from hrl.environment.game_env import GameEnvironment

def main():
    # Training configuration
    config = {
        'env_config': {
            'map_size': [100, 100],
            'num_agents': 3,
            'max_steps': 1000,
            'tag_radius': 5,
            'capture_radius': 10,
            'base_radius': 20,
            'difficulty': 0.5,
            'max_agents': 6,
            'num_flags': 2,
            'num_teams': 2
        },
        'policy_config': {
            'state_size': 8,  # Position[2], velocity[2], flags[1], tags[1], health[1], team[1]
            'action_size': 2,  # 2D movement
            'hidden_size': 128,
            'num_options': 6,  # Number of available options
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'lambda_': 0.95,
            'entropy_coef': 0.01,
            'options': ['attack_flag', 'capture_flag', 'guard_flag', 'evade', 'tag', 'retreat']  # String options
        },
        'training_config': {
            'num_episodes': 10000,
            'batch_size': 32,
            'update_frequency': 10,
            'log_interval': 100,
            'save_interval': 1000,
            'log_dir': 'logs',
            'render': False  # Add render flag
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
            'update_frequency': 100
        }
    }

    # Initialize trainer
    trainer = Trainer(config)
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_checkpoint('final_model.pth')
    print("Training completed. Model saved to final_model.pth")

if __name__ == '__main__':
    main() 