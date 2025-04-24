#!/usr/bin/env python
"""
Utilities for testing the HRL framework.

This module provides common functions and utilities for testing the HRL system,
including state mocking, environment setup, and result visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import torch
import os
from typing import Dict, Any, List, Tuple, Optional

from hrl.environment.game_env import GameEnvironment, GameState, Agent


def create_test_env(config: Optional[Dict[str, Any]] = None) -> GameEnvironment:
    """
    Create a test environment with default configuration.
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        GameEnvironment instance
    """
    default_config = {
        'map_size': [100, 100],
        'num_agents': 3,
        'max_steps': 500,
        'tag_radius': 5,
        'capture_radius': 10,
        'base_radius': 20,
        'debug_level': 0
    }
    
    if config:
        default_config.update(config)
        
    return GameEnvironment(default_config)


def create_test_state(env: GameEnvironment) -> Dict[str, Any]:
    """
    Create a basic test state from the environment.
    
    Args:
        env: Game environment
        
    Returns:
        State dictionary
    """
    state = env._get_observation()
    
    # Add agent IDs if not present
    for i, agent in enumerate(state['agents']):
        if 'id' not in agent:
            agent['id'] = i
            
    return state


def create_complex_state(env: GameEnvironment) -> Dict[str, Any]:
    """
    Create a more complex state with various game scenarios.
    
    Args:
        env: Game environment
        
    Returns:
        Complex state dictionary
    """
    # First create a base state
    state = create_test_state(env)
    
    # Modify it to create a more interesting scenario
    # 1. Move some agents to strategic positions
    for i, agent in enumerate(state['agents']):
        if agent['team'] == 0:  # Our team
            if i == 0:
                # First agent near enemy flag
                enemy_flag = next(flag for flag in state['flags'] if flag['team'] == 1)
                agent['position'] = enemy_flag['position'] - np.array([10, 0])
            elif i == 1:
                # Second agent in the middle
                agent['position'] = np.array(env.map_size) / 2
            else:
                # Others near our base
                our_base = state['team_bases'][0]
                agent['position'] = our_base + np.random.uniform(-15, 15, 2)
        else:  # Enemy team
            if i == 3:  # First enemy agent approaching our flag
                our_flag = next(flag for flag in state['flags'] if flag['team'] == 0)
                agent['position'] = our_flag['position'] + np.array([15, 5])
            elif i == 4:  # Second enemy agent has our flag
                our_flag = next(flag for flag in state['flags'] if flag['team'] == 0)
                our_flag['is_captured'] = True
                our_flag['carrier_id'] = agent['id']
                agent['has_flag'] = True
                agent['position'] = our_flag['position'] + np.array([20, 0])
            else:
                # Others randomly positioned
                agent['position'] = np.random.uniform([0, 0], env.map_size, 2)
    
    return state


def visualize_state(state: Dict[str, Any], title: str = "Game State Visualization"):
    """
    Visualize a game state.
    
    Args:
        state: Game state to visualize
        title: Title for the visualization
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    map_size = state.get('map_size', [100, 100])
    ax.set_xlim(0, map_size[0])
    ax.set_ylim(0, map_size[1])
    
    # Draw territories
    for team, territory in state['territories'].items():
        color = 'lightblue' if team == 0 else 'lightpink'
        polygon = Polygon(territory, alpha=0.3, color=color)
        ax.add_patch(polygon)
    
    # Draw bases
    for team, base_pos in state['team_bases'].items():
        color = 'blue' if team == 0 else 'red'
        base_circle = Circle(base_pos, 20, alpha=0.3, color=color)
        ax.add_patch(base_circle)
    
    # Draw flags
    for flag in state['flags']:
        color = 'blue' if flag['team'] == 0 else 'red'
        if not flag['is_captured']:
            ax.plot(flag['position'][0], flag['position'][1], 'v', markersize=12, color=color)
    
    # Draw agents
    for i, agent in enumerate(state['agents']):
        agent_id = agent.get('id', i)
        team = agent['team']
        position = agent['position']
        
        # Decide marker and color
        if team == 0:  # Our team
            color = 'blue'
            marker = 'o'
        else:  # Enemy team
            color = 'red'
            marker = 's'  # square for enemies
        
        # Make size larger for flag carriers
        size = 12 if agent.get('has_flag', False) else 8
        
        # Special marker for tagged agents
        if agent.get('is_tagged', False):
            marker = 'x'
        
        # Plot agent
        ax.plot(position[0], position[1], marker, color=color, markersize=size)
        
        # Add agent ID
        ax.text(position[0] + 2, position[1] + 2, str(agent_id), color='black')
    
    # Add title
    ax.set_title(title)
    
    return fig, ax


def load_model(model_path: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a trained model from a checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        
    Returns:
        model: Loaded model
        config: Model configuration
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    try:
        checkpoint = torch.load(model_path, weights_only=False)
        model = checkpoint.get('model', None)
        policy_state_dict = checkpoint.get('policy_state_dict', None)
        config = checkpoint.get('config', {})
        
        # Return model and config
        return model, config
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, {}


def compare_results(results_a: List[Dict[str, Any]], 
                   results_b: List[Dict[str, Any]], 
                   metric: str = 'win_rate',
                   title: str = "Results Comparison"):
    """
    Compare results from two different test runs.
    
    Args:
        results_a: First set of results
        results_b: Second set of results
        metric: Metric to compare
        title: Title for the plot
    """
    # Extract strategies and metric values
    strategies = [r["strategy"] for r in results_a]
    values_a = [r.get(metric, 0) for r in results_a]
    values_b = [r.get(metric, 0) for r in results_b]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up bars
    x = np.arange(len(strategies))
    width = 0.35
    
    # Plot bars
    ax.bar(x - width/2, values_a, width, label='Model A')
    ax.bar(x + width/2, values_b, width, label='Model B')
    
    # Add labels and title
    ax.set_xlabel('Strategy')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax


def print_test_summary(name: str, results: Dict[str, Any]):
    """
    Print a formatted summary of test results.
    
    Args:
        name: Name of the test
        results: Test results dictionary
    """
    print("\n" + "="*60)
    print(f"SUMMARY: {name}")
    print("="*60)
    
    # Print metrics
    for key, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{key.replace('_', ' ').title()}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"\n{key.replace('_', ' ').title()}:")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, (int, float)):
                    print(f"  {subkey.replace('_', ' ').title()}: {subvalue:.4f}")
                else:
                    print(f"  {subkey.replace('_', ' ').title()}: {subvalue}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
            
    print("="*60)


def run_random_episode(env: GameEnvironment, render: bool = False) -> Dict[str, Any]:
    """
    Run an episode with random actions and track metrics.
    
    Args:
        env: Game environment
        render: Whether to render the environment
        
    Returns:
        Dict with episode results
    """
    state = env.reset()
    done = False
    step = 0
    total_reward = 0
    
    # Track metrics
    metrics = {
        "steps": 0,
        "total_reward": 0,
        "flag_captures": 0,
        "tags": 0,
        "outcome": None
    }
    
    while not done:
        # Generate random actions for each agent
        actions = []
        for _ in range(len(env.agents)):
            # Random velocity in [-1, 1] range
            velocity = np.random.uniform(-1, 1, 2)
            actions.append(velocity)
            
        # Step environment
        next_state, reward, done, info = env.step(actions)
        
        # Update metrics
        total_reward += reward
        step += 1
        
        if info.get('flag_captured', False):
            metrics["flag_captures"] += 1
            
        if info.get('tagged', []):
            metrics["tags"] += len(info['tagged'])
            
        # Render if requested
        if render:
            env.render()
            
        # Update state
        state = next_state
        
    # Update final metrics
    metrics["steps"] = step
    metrics["total_reward"] = total_reward
    metrics["outcome"] = env.game_state.name
    
    return metrics 