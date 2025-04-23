#!/usr/bin/env python
"""
Helper script to run the environment and test a trained model.
"""

import argparse
import os
import torch
import numpy as np
from hrl.environment.game_env import GameEnvironment
from hrl.policies.hierarchical_policy import HierarchicalPolicy
from hrl.utils.option_selector import OptionSelector
from hrl.utils.state_processor import StateProcessor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test a trained model on the capture-the-flag environment.')
    
    parser.add_argument('--model', type=str, default='final_model',
                       help='Model to load (without extension)')
    parser.add_argument('--checkpoint-dir', type=str, default=os.path.join('hrl', 'checkpoints'),
                       help='Directory containing model checkpoints')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment')
    parser.add_argument('--debug-level', type=int, default=1,
                       help='Debug level (0=minimal, 1=normal, 2=verbose)')
    
    return parser.parse_args()

def load_model(model_path):
    """Load a trained model from a checkpoint file."""
    try:
        # Set weights_only=False to handle PyTorch 2.6 changes
        checkpoint = torch.load(model_path, weights_only=False)
        policy_config = checkpoint['config']['policy_config']
        env_config = checkpoint['config']['env_config']
        
        # Create policy and option selector
        policy = HierarchicalPolicy(policy_config)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        
        option_selector = OptionSelector(policy_config)
        option_selector.load_state_dict(checkpoint['option_selector_state_dict'])
        
        state_processor = StateProcessor(env_config)
        
        print(f"Successfully loaded model from {model_path}")
        return policy, option_selector, state_processor, env_config
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None

def run_episodes(policy, option_selector, state_processor, env_config, num_episodes=5, render=False, debug_level=1):
    """Run episodes with the loaded model."""
    if not policy or not option_selector or not state_processor:
        print("Cannot run episodes without a valid model")
        return
    
    # Update environment config with debug level
    env_config['debug_level'] = debug_level
    
    # Create environment
    env = GameEnvironment(env_config)
    
    # Track results
    wins = 0
    losses = 0
    draws = 0
    total_rewards = []
    
    print(f"\n{'='*60}")
    print(f"Running {num_episodes} test episodes")
    print(f"{'='*60}")
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        print(f"\nStarting Episode {episode+1}/{num_episodes}")
        print(f"{'-'*40}")
        
        while not done:
            # Process state
            processed_state = state_processor.process_state(state)
            
            # Select option
            option = option_selector.select_option(processed_state)
            
            # Get action from policy
            action = policy.get_action(processed_state, option)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Accumulate reward
            episode_reward += reward
            
            # Update state
            state = next_state
            
            # Render if enabled
            if render:
                env.render()
                
        # Print episode results
        print(f"Episode {episode+1} completed with reward: {episode_reward:.2f}")
        total_rewards.append(episode_reward)
        
        # Record outcome
        if env.game_state.name == 'WON':
            wins += 1
            print("Team 0 (Agent) WON!")
        elif env.game_state.name == 'LOST':
            losses += 1
            print("Team 1 (Opponent) WON!")
        else:
            draws += 1
            print("DRAW!")
            
        print(f"Final score: Team 0 (Agent) {env.team_scores[0]} - {env.team_scores[1]} Team 1 (Opponent)")
    
    # Close environment
    env.close()
    
    # Print summary
    print(f"\n{'='*60}")
    print("Test Summary:")
    print(f"Episodes: {num_episodes}")
    print(f"Wins: {wins} ({wins/num_episodes*100:.1f}%)")
    print(f"Losses: {losses} ({losses/num_episodes*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_episodes*100:.1f}%)")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"{'='*60}")

def main():
    """Main function."""
    args = parse_args()
    
    # Construct model path
    model_name = args.model
    if not model_name.endswith('.pth'):
        model_name += '.pth'
    model_path = os.path.join(args.checkpoint_dir, model_name)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    # Load model
    policy, option_selector, state_processor, env_config = load_model(model_path)
    
    # Run episodes
    run_episodes(
        policy=policy,
        option_selector=option_selector,
        state_processor=state_processor,
        env_config=env_config,
        num_episodes=args.episodes,
        render=args.render,
        debug_level=args.debug_level
    )

if __name__ == '__main__':
    main() 