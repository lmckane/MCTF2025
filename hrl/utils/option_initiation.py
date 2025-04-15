from typing import Dict, Any
import numpy as np

class OptionInitiation:
    """Handles initiation conditions for options."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initiation_threshold = config.get("initiation_threshold", 1.0)
        
    def can_initiate(self, state: Dict[str, Any], option_name: str) -> bool:
        """
        Check if an option can be initiated in the current state.
        
        Args:
            state: Current environment state
            option_name: Name of the option to check
            
        Returns:
            bool: Whether the option can be initiated
        """
        if option_name == "capture":
            return self._can_initiate_capture(state)
        elif option_name == "defend":
            return self._can_initiate_defend(state)
        elif option_name == "patrol":
            return self._can_initiate_patrol(state)
        return False
        
    def _can_initiate_capture(self, state: Dict[str, Any]) -> bool:
        """Check if capture option can be initiated."""
        # Can't capture if already has flag
        if state.get("has_flag", False):
            return False
            
        # Check if opponent's flag is reachable
        agent_pos = state["agent_position"]
        flag_pos = state["opponent_flag_position"]
        distance = np.linalg.norm(agent_pos - flag_pos)
        
        # Check if any opponents are too close
        for opp_pos in state.get("opponent_positions", []):
            opp_distance = np.linalg.norm(agent_pos - opp_pos)
            if opp_distance < self.initiation_threshold:
                return False
                
        return distance < self.initiation_threshold * 2
        
    def _can_initiate_defend(self, state: Dict[str, Any]) -> bool:
        """Check if defend option can be initiated."""
        # Check if own flag is in danger
        flag_pos = state["team_flag_position"]
        
        # Check if any opponents are near the flag
        for opp_pos in state.get("opponent_positions", []):
            distance = np.linalg.norm(flag_pos - opp_pos)
            if distance < self.initiation_threshold:
                return True
                
        return False
        
    def _can_initiate_patrol(self, state: Dict[str, Any]) -> bool:
        """Check if patrol option can be initiated."""
        # Can always patrol if no other options are available
        return True 