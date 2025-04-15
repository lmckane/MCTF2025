from typing import Dict, Any
import numpy as np
from hrl.policies.base import BaseHierarchicalPolicy

class MetaPolicy(BaseHierarchicalPolicy):
    """Meta-policy that selects between different options."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.option_values = {}  # Q-values for each option
        self.learning_rate = config.get("learning_rate", 0.01)
        self.discount_factor = config.get("discount_factor", 0.99)
        
    def get_option_score(self, state: Dict[str, Any], option_name: str) -> float:
        """Get the score for an option in the current state."""
        if option_name not in self.option_values:
            self.option_values[option_name] = {}
            
        state_key = self._get_state_key(state)
        if state_key not in self.option_values[option_name]:
            self.option_values[option_name][state_key] = 0.0
            
        return self.option_values[option_name][state_key]
        
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """Update the option values based on the transition."""
        if self.current_option is None:
            return
            
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Update Q-value for current option
        current_value = self.option_values[self.current_option].get(state_key, 0.0)
        next_value = self.option_values[self.current_option].get(next_state_key, 0.0)
        
        # Q-learning update
        target = reward + (1 - done) * self.discount_factor * next_value
        self.option_values[self.current_option][state_key] = (
            current_value + self.learning_rate * (target - current_value)
        )
        
    def _get_state_key(self, state: Dict[str, Any]) -> str:
        """Convert state to a string key for value storage."""
        # Create a simplified state representation
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
        
    def reset(self):
        """Reset the meta-policy's internal state."""
        super().reset()
        # Keep option values but reset current option
        self.current_option = None 