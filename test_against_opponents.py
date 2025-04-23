#!/usr/bin/env python
"""
Script to test our trained model against different opponent strategies.
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple

from hrl.environment.game_env import GameEnvironment, Agent, GameState
from hrl.policies.hierarchical_policy import HierarchicalPolicy
from hrl.utils.option_selector import OptionSelector
from hrl.utils.state_processor import StateProcessor
from hrl.environment.advanced_opponents import get_opponent_strategy, OPPONENT_STRATEGIES

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test a trained model against different opponent strategies.')
    
    parser.add_argument('--model', type=str, default='final_model',
                       help='Model to load (without extension)')
    parser.add_argument('--checkpoint-dir', type=str, default=os.path.join('hrl', 'checkpoints'),
                       help='Directory containing model checkpoints')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run per opponent strategy')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment')
    parser.add_argument('--opponent', type=str, default=None,
                       help='Specific opponent strategy to test against (default: test all)')
    parser.add_argument('--debug-level', type=int, default=1,
                       help='Debug level (0=minimal, 1=normal, 2=verbose)')
    parser.add_argument('--difficulty', type=float, default=0.8,
                       help='Difficulty level (0.0-1.0)')
    parser.add_argument('--output', type=str, default='opponent_results.png',
                       help='Output file for results chart')
    
    return parser.parse_args()

def load_model(model_path):
    """Load a trained model from a checkpoint file."""
    try:
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

def run_episode(policy, option_selector, state_processor, env_config, opponent_strategy, 
               render=False, debug_level=1) -> Tuple[bool, float, int, int]:
    """Run a single episode against an opponent strategy.
    
    Returns:
        win (bool): Whether the agent won
        reward (float): Total reward
        agent_score (int): Agent team score
        opponent_score (int): Opponent team score
    """
    # Update environment config
    env_config = env_config.copy()
    env_config['debug_level'] = debug_level
    
    # Create environment
    env = GameEnvironment(env_config)
    
    # Reset opponent strategy
    opponent_strategy.reset()
    
    # Initialize episode
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Process state
        processed_state = state_processor.process_state(state)
        
        # Select option and get action for our agent (Team 0, Agent 0)
        option = option_selector.select_option(processed_state)
        action = policy.get_action(processed_state, option)
        
        # The environment expects a single action, it handles actions for all agents internally
        # We'll override the opponent actions in the _get_opponent_action method
        
        # Take step in environment with our custom opponent actions
        next_state, reward, done, info = env.step(action)
        
        # Override opponent actions
        for agent in env.agents:
            if agent.team == 1:  # Opponent team
                opponent_action = opponent_strategy.get_action(agent, state)
                agent.velocity = opponent_action * env.max_velocity
                agent.position += agent.velocity
                agent.position = np.clip(agent.position, 0, env.map_size - 1)
        
        # Accumulate reward
        total_reward += reward
        
        # Update state
        state = next_state
        
        # Render if enabled
        if render:
            env.render()
    
    # Get final scores
    agent_score = env.team_scores[0]
    opponent_score = env.team_scores[1]
    
    # Determine win/loss
    win = (env.game_state == GameState.WON)
    
    # Close environment
    env.close()
    
    return win, total_reward, agent_score, opponent_score

def test_against_strategy(policy, option_selector, state_processor, env_config, 
                         strategy_name, episodes=5, render=False, debug_level=1, difficulty=0.8):
    """Test model against a specific opponent strategy."""
    print(f"\n{'='*60}")
    print(f"Testing against {strategy_name.upper()} strategy")
    print(f"{'='*60}")
    
    # Create opponent strategy
    strategy_config = {"difficulty": difficulty}
    opponent_strategy = get_opponent_strategy(strategy_name, strategy_config)
    
    # Results tracking
    wins = 0
    total_reward = 0
    team_scores = []
    opponent_scores = []
    
    for episode in range(episodes):
        print(f"\nEpisode {episode+1}/{episodes}")
        print(f"{'-'*40}")
        
        # Run episode
        win, reward, agent_score, opponent_score = run_episode(
            policy, option_selector, state_processor, env_config,
            opponent_strategy, render, debug_level
        )
        
        # Update stats
        wins += 1 if win else 0
        total_reward += reward
        team_scores.append(agent_score)
        opponent_scores.append(opponent_score)
        
        # Print episode results
        print(f"Episode {episode+1} completed with reward: {reward:.2f}")
        print(f"{'WIN' if win else 'LOSS'}")
        print(f"Score: Team 0 (Agent) {agent_score} - {opponent_score} Team 1 (Opponent)")
    
    # Calculate summary statistics
    win_rate = wins / episodes
    avg_reward = total_reward / episodes
    avg_score = sum(team_scores) / episodes
    avg_opponent_score = sum(opponent_scores) / episodes
    
    # Print summary
    print(f"\n{'-'*60}")
    print(f"Summary against {strategy_name.upper()} strategy:")
    print(f"Win Rate: {win_rate:.2f} ({wins}/{episodes})")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Score: {avg_score:.2f} - {avg_opponent_score:.2f}")
    print(f"{'-'*60}")
    
    return {
        "strategy": strategy_name,
        "win_rate": win_rate,
        "avg_reward": avg_reward,
        "avg_score": avg_score,
        "avg_opponent_score": avg_opponent_score,
        "wins": wins,
        "episodes": episodes
    }

def plot_results(results, output_file="opponent_results.png"):
    """Plot the results of opponent testing."""
    strategies = [r["strategy"] for r in results]
    win_rates = [r["win_rate"] for r in results]
    avg_rewards = [r["avg_reward"] for r in results]
    avg_scores = [r["avg_score"] for r in results]
    avg_opp_scores = [r["avg_opponent_score"] for r in results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot win rates
    axes[0].bar(strategies, win_rates, color='blue')
    axes[0].set_ylim(0, 1)
    axes[0].set_title('Win Rate by Opponent Strategy')
    axes[0].set_ylabel('Win Rate')
    
    # Plot average rewards
    axes[1].bar(strategies, avg_rewards, color='green')
    axes[1].set_title('Average Reward by Opponent Strategy')
    axes[1].set_ylabel('Avg Reward')
    
    # Plot average scores
    x = np.arange(len(strategies))
    width = 0.35
    axes[2].bar(x - width/2, avg_scores, width, label='Agent Score')
    axes[2].bar(x + width/2, avg_opp_scores, width, label='Opponent Score')
    axes[2].set_title('Average Scores by Opponent Strategy')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(strategies)
    axes[2].set_ylabel('Avg Score')
    axes[2].legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Results chart saved to {output_file}")

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
    if not policy:
        return
    
    # Update difficulty
    env_config['difficulty'] = args.difficulty
    
    # Determine strategies to test
    if args.opponent:
        if args.opponent not in OPPONENT_STRATEGIES:
            print(f"Opponent strategy '{args.opponent}' not found. Available strategies:")
            for strategy in OPPONENT_STRATEGIES:
                print(f"  - {strategy}")
            return
        strategies_to_test = [args.opponent]
    else:
        strategies_to_test = list(OPPONENT_STRATEGIES.keys())
    
    # Test against each strategy
    results = []
    for strategy in strategies_to_test:
        result = test_against_strategy(
            policy, option_selector, state_processor, env_config,
            strategy, args.episodes, args.render, args.debug_level, args.difficulty
        )
        results.append(result)
    
    # Print overall summary
    print("\n" + "="*60)
    print("OVERALL RESULTS")
    print("="*60)
    print(f"{'Strategy':<12} {'Win Rate':<10} {'Avg Reward':<12} {'Avg Score':<10}")
    print("-"*60)
    for result in results:
        print(f"{result['strategy']:<12} {result['win_rate']:.2f} ({result['wins']}/{result['episodes']}) "
              f"{result['avg_reward']:<12.2f} {result['avg_score']:.1f}-{result['avg_opponent_score']:.1f}")
    
    # Plot results
    if len(results) > 1:
        plot_results(results, args.output)

if __name__ == '__main__':
    main() 