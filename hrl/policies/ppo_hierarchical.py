import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
import numpy as np
from typing import Dict, Any, List
import gymnasium as gym
from pyquaticus.envs.pyquaticus import PyQuaticusEnv

from hrl.policies.base import BaseHierarchicalPolicy

class PPOHierarchicalPolicy(BaseHierarchicalPolicy):
    """Hierarchical policy using PPO for high-level option selection."""
    
    def __init__(self, options: List[str], config: Dict[str, Any]):
        """
        Initialize the PPO hierarchical policy.
        
        Args:
            options: List of available high-level behaviors/options
            config: Configuration dictionary for PPO
        """
        super().__init__(options)
        self.config = config
        
        # Initialize PPO trainer with updated API
        ppo_config = PPOConfig()
        ppo_config = ppo_config.api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        ).training(
            gamma=config.get("gamma", 0.99),
            lr=config.get("lr", 0.0001),
            train_batch_size=config.get("train_batch_size", 4000),
            num_sgd_iter=config.get("num_sgd_iter", 10),
            clip_param=config.get("clip_param", 0.2),
            lambda_=config.get("lambda", 0.95),
            model=config.get("model", {
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "tanh"
            })
        )
        
        self.trainer = ppo_config.build()
        
    def select_option(self, state: Dict[str, Any]) -> str:
        """
        Select the next high-level option using PPO.
        
        Args:
            state: Current environment state
            
        Returns:
            str: Selected option name
        """
        # Convert state to observation format expected by PPO
        obs = self._process_state(state)
        
        # Get action from PPO
        action = self.trainer.compute_single_action(obs)
        
        # Map action to option
        self.current_option = self.options[action]
        self.option_history.append(self.current_option)
        
        return self.current_option
    
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get the action to take based on the current state and selected option.
        This is a placeholder as the actual action will be determined by the selected option.
        
        Args:
            state: Current environment state
            
        Returns:
            np.ndarray: Action to take
        """
        # The actual action will be determined by the selected option
        return np.zeros(2)  # Placeholder
    
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """
        Update the PPO policy based on the transition.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Convert states to observation format
        obs = self._process_state(state)
        next_obs = self._process_state(next_state)
        
        # Update PPO
        self.trainer.train()
        
    def _process_state(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Process the state dictionary into a format suitable for PPO.
        
        Args:
            state: Environment state dictionary
            
        Returns:
            np.ndarray: Processed observation
        """
        # Extract relevant features from state
        features = []
        
        # Add agent position
        if 'agent_position' in state:
            features.extend(state['agent_position'])
            
        # Add flag positions
        if 'flag_position' in state:
            features.extend(state['flag_position'].flatten())
            
        # Add opponent positions
        if 'opponent_position' in state:
            features.extend(state['opponent_position'].flatten())
            
        # Add agent status
        if 'agent_has_flag' in state:
            features.append(float(state['agent_has_flag']))
        if 'agent_is_tagged' in state:
            features.append(float(state['agent_is_tagged']))
            
        return np.array(features, dtype=np.float32)
    
    def save(self, path: str):
        """Save the trained policy."""
        self.trainer.save(path)
        
    def load(self, path: str):
        """Load a trained policy."""
        self.trainer.restore(path)

    def train(self, env: PyQuaticusEnv, num_episodes: int = 1000):
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                self.trainer.compute_single_action(
                    observation=state,
                    prev_action=action,
                    prev_reward=reward,
                    info=info
                )
                state = next_state
                episode_reward += reward
                
            print(f"Episode {episode + 1}: Reward = {episode_reward}")
            
    def select_action(self, state: np.ndarray) -> np.ndarray:
        action = self.trainer.compute_single_action(state)
        return action 