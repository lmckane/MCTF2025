from typing import Dict, Any
import numpy as np

class OptionTermination:
    """Handles termination conditions for options."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.termination_threshold = config.get("termination_threshold", 0.1)
        self.max_steps = config.get("max_steps", 100)
        self.current_steps = 0
        
    def should_terminate(self, state: Dict[str, Any], option_name: str) -> bool:
        """
        Check if the current option should terminate.
        
        Args:
            state: Current environment state
            option_name: Name of the current option
            
        Returns:
            bool: Whether the option should terminate
        """
        self.current_steps += 1
        
        # Check max steps
        if self.current_steps >= self.max_steps:
            return True
            
        # Check goal achievement
        if self._check_goal_achievement(state, option_name):
            return True
            
        # Check failure conditions
        if self._check_failure_conditions(state, option_name):
            return True
            
        return False
        
    def _check_goal_achievement(self, state: Dict[str, Any], option_name: str) -> bool:
        """Check if the option's goal has been achieved."""
        if option_name == "capture":
            # Check if agent has reached opponent's flag
            agent_pos = state["agent_position"]
            flag_pos = state["opponent_flag_position"]
            distance = np.linalg.norm(agent_pos - flag_pos)
            return distance < self.termination_threshold
            
        elif option_name == "defend":
            # Check if agent is close enough to own flag
            agent_pos = state["agent_position"]
            flag_pos = state["team_flag_position"]
            distance = np.linalg.norm(agent_pos - flag_pos)
            return distance < self.termination_threshold
            
        elif option_name == "patrol":
            # Check if agent has reached patrol point
            agent_pos = state["agent_position"]
            patrol_pos = state.get("patrol_point", agent_pos)
            distance = np.linalg.norm(agent_pos - patrol_pos)
            return distance < self.termination_threshold
            
        return False
        
    def _check_failure_conditions(self, state: Dict[str, Any], option_name: str) -> bool:
        """Check if the option has failed."""
        # Check if agent is tagged
        if state.get("is_tagged", False):
            return True
            
        # Check if agent is out of bounds
        if not self._is_in_bounds(state["agent_position"]):
            return True
            
        return False
        
    def _is_in_bounds(self, position: np.ndarray) -> bool:
        """Check if position is within environment bounds."""
        bounds = self.config.get("env_bounds", [[-10, 10], [-10, 10]])
        return (bounds[0][0] <= position[0] <= bounds[0][1] and 
                bounds[1][0] <= position[1] <= bounds[1][1])
                
    def reset(self):
        """Reset the termination checker's state."""
        self.current_steps = 0 