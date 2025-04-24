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

# Add the imports for Pyquaticus
from pyquaticus import pyquaticus_v0
from pyquaticus.envs.pyquaticus import Team
import pyquaticus.config
import pyquaticus.utils.rewards as rewards

from hrl.environment.game_env import GameEnvironment, Agent, GameState
from hrl.environment.pyquaticus_wrapper import PyquaticusWrapper
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
    
    # Create Pyquaticus environment directly
    pyq_config = pyquaticus.config.config_dict_std.copy()
    pyq_config['max_time'] = env_config.get('max_steps', 1000)
    pyq_config['max_score'] = 3  # Max score needed to win
    
    # IMPORTANT: In Pyquaticus, catch_radius MUST be greater than flag_keepout
    # Default values in Pyquaticus std config are:
    # catch_radius = 10.0, flag_keepout = 3.0
    # So we'll maintain that ratio
    pyq_config['catch_radius'] = 10.0  # Use Pyquaticus default
    pyq_config['flag_keepout'] = 3.0   # Use Pyquaticus default
    pyq_config['flag_radius'] = 2.0    # Use Pyquaticus default
    
    pyq_config['tagging_cooldown'] = 60
    pyq_config['tag_on_oob'] = True
    pyq_config['sim_speedup_factor'] = 4
    
    # Set up rewards for our team only
    reward_config = {}
    for i in range(3):  # 3 agents per team
        reward_config[f'agent_{i}'] = rewards.caps_and_grabs
    for i in range(3, 6):
        reward_config[f'agent_{i}'] = None
    
    # Determine rendering mode
    render_mode = 'human' if render else None
    
    # Create environment
    env = pyquaticus_v0.PyQuaticusEnv(
        config_dict=pyq_config,
        render_mode=render_mode,
        reward_config=reward_config,
        team_size=3
    )
    
    # Reset opponent strategy
    opponent_strategy.reset()
    
    # Initialize episode
    env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    # Create PyquaticusWrapper for observation formatting
    wrapper = PyquaticusWrapper(env_config)
    wrapper.env = env
    
    # Get initial observation
    state = wrapper._format_observation(env.observe('agent_0'))
    
    while not done:
        # Process state
        processed_state = state_processor.process_state(state)
        
        # DEBUG: Print agent positions and roles every 50 steps
        if debug_level >= 2 and step_count % 50 == 0:
            print("\nAgent Positions Debug:")
            for i, agent in enumerate(env.par_env.env.agents):
                team_name = "Ally" if i < 3 else "Enemy"
                agent_pos = agent.position
                agent_has_flag = "HAS FLAG" if agent.has_flag else ""
                agent_tagged = "TAGGED" if agent.is_tagged else ""
                
                # Get agent role for ally agents
                agent_role = "UNKNOWN"
                if i < 3:
                    # Try to get agent role from processed state
                    if hasattr(processed_state, 'agent_roles') and i < len(processed_state.agent_roles):
                        role_val = processed_state.agent_roles[i]
                        if role_val == 0:
                            agent_role = "ATTACKER"
                        elif role_val == 1:
                            agent_role = "DEFENDER"
                        elif role_val == 2:
                            agent_role = "INTERCEPTOR"
                
                print(f"  {team_name} Agent {i}: Pos={agent_pos} Role={agent_role} {agent_has_flag} {agent_tagged}")
            
            # Print map boundaries
            map_size = env.par_env.env.map_size
            print(f"  Map Size: {map_size}")
            print(f"  Team Bases: Team BLUE={env.par_env.env.team_bases[0]}, Team RED={env.par_env.env.team_bases[1]}")
            print(f"  Flag Positions: Team BLUE={env.par_env.env.flags[0].position}, Team RED={env.par_env.env.flags[1].position}")
            
            # Check for ally agents potentially stuck in enemy territory
            enemy_base = env.par_env.env.team_bases[1]
            for i in range(3):  # Ally agents (0-2)
                agent = env.par_env.env.agents[i]
                # Calculate distance to enemy base
                distance_to_enemy_base = np.linalg.norm(agent.position - enemy_base)
                
                # Check if agent is deep in enemy territory
                # Use the average map size for the threshold calculation
                map_size_avg = np.mean(env.par_env.env.map_size)
                if distance_to_enemy_base < map_size_avg * 0.2:  # Within 20% of map size to enemy base
                    print(f"  WARNING: Ally Agent {i} appears to be deep in enemy territory!")
                    print(f"           Distance to enemy base: {distance_to_enemy_base:.2f}")
                    
                    # Calculate distance to ally base
                    ally_base = env.par_env.env.team_bases[0]
                    distance_to_ally_base = np.linalg.norm(agent.position - ally_base)
                    print(f"           Distance to ally base: {distance_to_ally_base:.2f}")
        
        # Select option and get action for our agent (Team 0, Agent 0)
        option = option_selector.select_option(processed_state)
        action = policy.get_action(processed_state, option)
        
        # Convert continuous action to discrete Pyquaticus action
        pyq_action = wrapper._format_action(action)
        
        # Create action dictionary for all agents
        action_dict = {'agent_0': pyq_action}
        
        # Step the environment
        env.step(action_dict)
        
        # Override opponent actions using opponent strategy
        for i in range(3, 6):  # Opponent agents (3-5)
            agent_id = f'agent_{i}'
            agent = env.par_env.env.agents[i]
            
            # Create a copy of the state for opponent strategy
            agent_state = state.copy()
            
            # Add agent ID for the opponent strategy
            for j, agent_data in enumerate(agent_state['agents']):
                if 'id' not in agent_data:
                    agent_data['id'] = j
            
            # Get opponent action
            opponent_discrete_action = opponent_strategy.get_action(agent, agent_state)
            
            # Apply the discrete action to Pyquaticus
            action_dict[agent_id] = opponent_discrete_action
        
        # Update rewards
        reward = env.get_rewards().get('agent_0', 0.0)
        total_reward += reward
        
        # Check if episode is done
        done = env.terminations.get('agent_0', False)
        step_count += 1
        
        # Get observation for next step
        state = wrapper._format_observation(env.observe('agent_0'))
        
        # Render if enabled
        if render and env.render_mode is not None:
            env.render()
        
        # End if max steps reached
        if step_count >= pyq_config['max_time']:
            done = True
    
    # Get final scores
    agent_score = env.par_env.env.blue_score
    opponent_score = env.par_env.env.red_score
    
    # Determine win/loss
    win = agent_score > opponent_score
    
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