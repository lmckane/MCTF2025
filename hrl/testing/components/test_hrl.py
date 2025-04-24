import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import gymnasium as gym
from pyquaticus.envs.pyquaticus import PyQuaticusEnv
from typing import Dict, Any, List
import time

from hrl.policies.ppo_hierarchical import PPOHierarchicalPolicy
from hrl.options.attack_flag import AttackFlagOption
from hrl.options.guard_flag import GuardFlagOption
from hrl.utils.reward_shaping import RewardShaper

class HRLEvaluator:
    """Evaluator for testing and visualizing HRL performance."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HRL evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.env = PyQuaticusEnv(
            render_mode="human" if config.get("render", False) else None,
            **config.get("env_config", {})
        )
        
        # Initialize components
        self.policy = PPOHierarchicalPolicy(
            options=config["options"],
            config=config.get("policy_config", {})
        )
        
        self.options = {
            "attack_flag": AttackFlagOption(config.get("attack_config", {})),
            "guard_flag": GuardFlagOption(config.get("guard_config", {})),
        }
        
        self.reward_shaper = RewardShaper(config)
        
        # Initialize metrics
        self.metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "option_usage": {opt: 0 for opt in config["options"]},
            "flag_grabs": 0,
            "flag_captures": 0,
            "tags": 0,
        }
        
    def test_option(self, option_name: str, num_episodes: int = 10):
        """
        Test a specific option in isolation.
        
        Args:
            option_name: Name of the option to test
            num_episodes: Number of episodes to run
        """
        print(f"\nTesting option: {option_name}")
        
        for episode in range(num_episodes):
            state = self._reset_environment()
            done = False
            total_reward = 0
            episode_length = 0
            
            while not done:
                # Get action from the specified option
                action = self.options[option_name].get_action(state)
                
                # Take action and get next state
                next_state, reward, done, info = self._step_environment(action)
                
                # Update metrics
                total_reward += reward
                episode_length += 1
                
                # Check if option should terminate
                if self.options[option_name].is_termination_condition_met(next_state):
                    break
                    
                state = next_state
                
            print(f"Episode {episode + 1}: Reward = {total_reward}, Length = {episode_length}")
            
    def test_full_system(self, num_episodes: int = 10):
        """
        Test the full HRL system.
        
        Args:
            num_episodes: Number of episodes to run
        """
        print("\nTesting full HRL system")
        
        for episode in range(num_episodes):
            state = self._reset_environment()
            done = False
            total_reward = 0
            episode_length = 0
            current_option = None
            
            while not done:
                # Select high-level option
                option = self.policy.select_option(state)
                if option != current_option:
                    print(f"Switching to option: {option}")
                    current_option = option
                    self.metrics["option_usage"][option] += 1
                
                # Get action from selected option
                action = self.options[option].get_action(state)
                
                # Take action and get next state
                next_state, reward, done, info = self._step_environment(action)
                
                # Update metrics
                total_reward += reward
                episode_length += 1
                
                # Update game-specific metrics
                if info.get("flag_grabbed", False):
                    self.metrics["flag_grabs"] += 1
                if info.get("flag_captured", False):
                    self.metrics["flag_captures"] += 1
                if info.get("tag_made", False):
                    self.metrics["tags"] += 1
                
                # Check if option should terminate
                if self.options[option].is_termination_condition_met(next_state):
                    current_option = None
                    
                state = next_state
                
            self.metrics["episode_rewards"].append(total_reward)
            self.metrics["episode_lengths"].append(episode_length)
            
            print(f"Episode {episode + 1}: Reward = {total_reward}, Length = {episode_length}")
            
    def visualize_metrics(self):
        """Visualize the collected metrics."""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot episode rewards
        axes[0, 0].plot(self.metrics["episode_rewards"])
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Total Reward")
        
        # Plot episode lengths
        axes[0, 1].plot(self.metrics["episode_lengths"])
        axes[0, 1].set_title("Episode Lengths")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Steps")
        
        # Plot option usage
        options = list(self.metrics["option_usage"].keys())
        usage = list(self.metrics["option_usage"].values())
        axes[1, 0].bar(options, usage)
        axes[1, 0].set_title("Option Usage")
        axes[1, 0].set_xlabel("Option")
        axes[1, 0].set_ylabel("Count")
        
        # Plot game metrics
        game_metrics = ["flag_grabs", "flag_captures", "tags"]
        values = [self.metrics[metric] for metric in game_metrics]
        axes[1, 1].bar(game_metrics, values)
        axes[1, 1].set_title("Game Metrics")
        axes[1, 1].set_xlabel("Metric")
        axes[1, 1].set_ylabel("Count")
        
        plt.tight_layout()
        plt.show()
        
    def _reset_environment(self) -> Dict[str, Any]:
        """Reset the environment and return initial state."""
        obs, info = self.env.reset()
        return self._process_observation(obs)
        
    def _step_environment(self, action: np.ndarray) -> tuple:
        """Take a step in the environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        next_state = self._process_observation(obs)
        return next_state, reward, done, info
        
    def _process_observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Process the environment observation into our state format."""
        state = {}
        
        # Process agent information
        state['agent_position'] = obs['agent_position']
        state['agent_heading'] = obs['agent_heading']
        state['agent_speed'] = obs['agent_speed']
        state['agent_has_flag'] = obs['agent_has_flag']
        state['agent_is_tagged'] = obs['agent_is_tagged']
        
        # Process flag information
        state['flag_position'] = obs.get('flag_position', [0, 0])
        state['flag_is_captured'] = obs.get('flag_is_captured', False)
        
        # Process opponent information
        state['opponent_positions'] = obs.get('opponent_positions', [])
        state['opponent_headings'] = obs.get('opponent_headings', [])
        state['opponent_has_flags'] = obs.get('opponent_has_flags', [])
        
        # Process territory information
        state['in_own_territory'] = obs.get('in_own_territory', True)
        state['base_position'] = obs.get('base_position', [0, 0])
        
        # Add LIDAR data if available
        if 'lidar' in obs:
            state['lidar'] = obs['lidar']
            
        return state
        
    def record_episode(self):
        """Record a full episode for visualization."""
        state = self._reset_environment()
        done = False
        states = [state.copy()]
        options_executed = []
        current_option = None
        
        while not done:
            # Select high-level option
            option = self.policy.select_option(state)
            if option != current_option:
                current_option = option
                
            options_executed.append(option)
            
            # Get action from selected option
            action = self.options[option].get_action(state)
            
            # Take action and get next state
            next_state, reward, done, info = self._step_environment(action)
            
            # Store state
            states.append(next_state.copy())
            
            # Check if option should terminate
            if self.options[option].is_termination_condition_met(next_state):
                current_option = None
                
            state = next_state
            
        return states, options_executed 