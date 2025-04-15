import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import pretty_print
import numpy as np
from typing import Dict, Any
import gymnasium as gym
from pyquaticus.envs.pyquaticus import PyQuaticusEnv

from hrl.policies.base import BaseHierarchicalPolicy
from hrl.policies.ppo_hierarchical import PPOHierarchicalPolicy
from hrl.options.base import BaseOption
from hrl.options.attack_flag import AttackFlagOption
from hrl.options.guard_flag import GuardFlagOption
from hrl.utils.reward_shaping import RewardShaper

class HRLTrainer:
    """Trainer for hierarchical reinforcement learning."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HRL trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.reward_shaper = RewardShaper(config)
        
        # Initialize PyQuaticus environment
        self.env = PyQuaticusEnv(
            render_mode="human" if config.get("render", False) else None,
            **config.get("env_config", {})
        )
        
        # Initialize Ray
        ray.init()
        
        # Create the hierarchical policy
        self.policy = self._create_policy()
        
        # Create the options
        self.options = self._create_options()
        
    def _create_policy(self) -> BaseHierarchicalPolicy:
        """Create the hierarchical policy."""
        return PPOHierarchicalPolicy(
            options=self.config["options"],
            config=self.config.get("policy_config", {})
        )
        
    def _create_options(self) -> Dict[str, BaseOption]:
        """Create the low-level options."""
        options = {}
        
        # Create attack flag option
        options["attack_flag"] = AttackFlagOption(
            config=self.config.get("attack_config", {})
        )
        
        # Create guard flag option
        options["guard_flag"] = GuardFlagOption(
            config=self.config.get("guard_config", {})
        )
        
        # Add other options here as they are implemented
        # options["evade_opponent"] = EvadeOpponentOption(...)
        # options["tag_intruder"] = TagIntruderOption(...)
        # options["retreat_home"] = RetreatHomeOption(...)
        
        return options
        
    def train(self, num_episodes: int):
        """
        Train the hierarchical policy.
        
        Args:
            num_episodes: Number of episodes to train for
        """
        for episode in range(num_episodes):
            state = self._reset_environment()
            done = False
            total_reward = 0
            episode_length = 0
            
            while not done:
                # Select high-level option
                option = self.policy.select_option(state)
                
                # Get action from selected option
                action = self.options[option].get_action(state)
                
                # Take action and get next state
                next_state, reward, done, info = self._step_environment(action)
                
                # Shape the reward
                shaped_reward = self.reward_shaper.shape_reward(state, action, reward)
                
                # Update the policy and options
                self.policy.update(state, action, shaped_reward, next_state, done)
                self.options[option].update(state, action, shaped_reward, next_state, done)
                
                # Update state and total reward
                state = next_state
                total_reward += shaped_reward
                episode_length += 1
                
                # Check if option should terminate
                if self.options[option].is_termination_condition_met(state):
                    self.policy.current_option = None
                    
            print(f"Episode {episode + 1}: Total Reward = {total_reward}, Length = {episode_length}")
            
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
        state['team_flag_position'] = obs['team_flag_position']
        state['opponent_flag_position'] = obs['opponent_flag_position']
        state['flag_taken'] = obs['flag_taken']
        
        # Process opponent information
        state['opponent_position'] = obs['opponent_position']
        state['opponent_heading'] = obs['opponent_heading']
        state['opponent_speed'] = obs['opponent_speed']
        
        # Add environment information
        state['env_size'] = self.env.env_size
        
        return state
        
    def save(self, path: str):
        """Save the trained policy and options."""
        self.policy.save(path)
        # Save options if they have state to save
        
    def load(self, path: str):
        """Load a trained policy and options."""
        self.policy.load(path)
        # Load options if they have state to load

if __name__ == "__main__":
    # Example configuration
    config = {
        "render": True,
        "env_config": {
            "env_bounds": [160.0, 80.0],
            "agent_radius": 2.0,
            "flag_radius": 2.0,
            "catch_radius": 10.0,
        },
        "options": ["attack_flag", "guard_flag", "evade_opponent", "tag_intruder", "retreat_home"],
        "policy_config": {
            "train_batch_size": 4000,
            "sgd_minibatch_size": 128,
            "num_sgd_iter": 10,
            "lr": 3e-4,
        },
        "attack_config": {
            "max_stuck_steps": 50,
            "evade_distance": 10.0,
        },
        "guard_config": {
            "patrol_radius": 10.0,
            "num_patrol_points": 4,
            "intercept_distance": 20.0,
            "guard_radius": 30.0,
        },
        "max_loitering_steps": 50,
        "max_flag_distance": 100.0,
        "collision_threshold": 5.0,
    }
    
    # Create and train the HRL agent
    trainer = HRLTrainer(config)
    trainer.train(num_episodes=1000) 