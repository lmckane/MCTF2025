from typing import Dict, Any, Tuple
import numpy as np
from hrl.policies.base import BaseHierarchicalPolicy

class OptionPolicy(BaseHierarchicalPolicy):
    """Policy for executing a specific option."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.option_name = config.get("option_name", "unknown")
        self.learning_rate = config.get("learning_rate", 0.01)
        self.discount_factor = config.get("discount_factor", 0.99)
        self.action_values = {}  # Q-values for state-action pairs
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """Get the action to take in the current state."""
        state_key = self._get_state_key(state)
        
        # Initialize Q-values for this state if not seen before
        if state_key not in self.action_values:
            self.action_values[state_key] = {}
            
        # Get available actions
        available_actions = self._get_available_actions(state)
        
        # Select action with highest Q-value
        best_action = None
        best_value = float('-inf')
        
        for action in available_actions:
            action_key = self._get_action_key(action)
            if action_key not in self.action_values[state_key]:
                self.action_values[state_key][action_key] = 0.0
                
            value = self.action_values[state_key][action_key]
            if value > best_value:
                best_value = value
                best_action = action
                
        return best_action if best_action is not None else np.zeros(2)
        
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """Update the policy based on the transition."""
        state_key = self._get_state_key(state)
        action_key = self._get_action_key(action)
        next_state_key = self._get_state_key(next_state)
        
        # Initialize Q-values if needed
        if state_key not in self.action_values:
            self.action_values[state_key] = {}
        if action_key not in self.action_values[state_key]:
            self.action_values[state_key][action_key] = 0.0
            
        # Get next state's best action value
        next_value = 0.0
        if not done and next_state_key in self.action_values:
            next_value = max(self.action_values[next_state_key].values(), default=0.0)
            
        # Q-learning update
        target = reward + (1 - done) * self.discount_factor * next_value
        self.action_values[state_key][action_key] = (
            self.action_values[state_key][action_key] + 
            self.learning_rate * (target - self.action_values[state_key][action_key])
        )
        
    def _get_state_key(self, state: Dict[str, Any]) -> str:
        """Convert state to a string key for value storage."""
        key_parts = []
        
        # Add agent position (rounded to 1 decimal place)
        if "agent_position" in state:
            pos = state["agent_position"]
            key_parts.append(f"pos_{pos[0]:.1f}_{pos[1]:.1f}")
            
        # Add flag positions
        if "team_flag_position" in state:
            pos = state["team_flag_position"]
            key_parts.append(f"team_flag_{pos[0]:.1f}_{pos[1]:.1f}")
            
        if "opponent_flag_position" in state:
            pos = state["opponent_flag_position"]
            key_parts.append(f"opp_flag_{pos[0]:.1f}_{pos[1]:.1f}")
            
        # Add opponent positions
        if "opponent_positions" in state:
            for i, pos in enumerate(state["opponent_positions"]):
                key_parts.append(f"opp_{i}_{pos[0]:.1f}_{pos[1]:.1f}")
                
        return "_".join(key_parts)
        
    def _get_action_key(self, action: np.ndarray) -> str:
        """Convert action to a string key for value storage."""
        return f"action_{action[0]:.1f}_{action[1]:.1f}"
        
    def _get_available_actions(self, state: Dict[str, Any]) -> list:
        """Get list of available actions in the current state."""
        # Default to 8 cardinal directions plus no-op
        actions = [
            np.array([1, 0]),   # right
            np.array([1, 1]),   # right-up
            np.array([0, 1]),   # up
            np.array([-1, 1]),  # left-up
            np.array([-1, 0]),  # left
            np.array([-1, -1]), # left-down
            np.array([0, -1]),  # down
            np.array([1, -1]),  # right-down
            np.array([0, 0])    # no-op
        ]
        return actions
        
    def reset(self):
        """Reset the policy's internal state."""
        super().reset()
        # Keep action values but reset current state
        self.current_state = None 