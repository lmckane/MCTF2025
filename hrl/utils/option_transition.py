from typing import Dict, Any, Tuple
import numpy as np

class OptionTransition:
    """Handles state transitions within options."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transition_noise = config.get("transition_noise", 0.1)
        
    def get_next_state(self, state: Dict[str, Any], action: np.ndarray, option_name: str) -> Dict[str, Any]:
        """
        Predict the next state given current state and action.
        
        Args:
            state: Current environment state
            action: Action to take
            option_name: Name of the current option
            
        Returns:
            Dict[str, Any]: Predicted next state
        """
        next_state = state.copy()
        
        # Update agent position with noise
        agent_pos = state["agent_position"]
        next_pos = agent_pos + action + np.random.normal(0, self.transition_noise, 2)
        next_state["agent_position"] = next_pos
        
        # Update opponent positions (simplified model)
        if "opponent_positions" in state:
            next_state["opponent_positions"] = self._update_opponent_positions(
                state["opponent_positions"],
                next_pos
            )
            
        # Update flag positions if they're being carried
        if state.get("has_flag", False):
            next_state["opponent_flag_position"] = next_pos
            
        return next_state
        
    def _update_opponent_positions(self, opponent_positions: np.ndarray, agent_pos: np.ndarray) -> np.ndarray:
        """Update opponent positions based on agent's movement."""
        next_positions = opponent_positions.copy()
        
        # Simple model: opponents move towards agent with some noise
        for i in range(len(opponent_positions)):
            direction = agent_pos - opponent_positions[i]
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction = direction / distance
                next_positions[i] += direction * 0.1 + np.random.normal(0, self.transition_noise, 2)
                
        return next_positions
        
    def get_transition_probability(self, state: Dict[str, Any], action: np.ndarray, 
                                 next_state: Dict[str, Any], option_name: str) -> float:
        """
        Get the probability of transitioning to next_state from state with action.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            option_name: Name of the current option
            
        Returns:
            float: Transition probability
        """
        # Simplified model: probability based on distance between predicted and actual next state
        predicted_state = self.get_next_state(state, action, option_name)
        
        # Compare agent positions
        pred_pos = predicted_state["agent_position"]
        actual_pos = next_state["agent_position"]
        pos_diff = np.linalg.norm(pred_pos - actual_pos)
        
        # Compare opponent positions
        opp_diff = 0
        if "opponent_positions" in state:
            pred_opp = predicted_state["opponent_positions"]
            actual_opp = next_state["opponent_positions"]
            opp_diff = np.mean([np.linalg.norm(p1 - p2) 
                              for p1, p2 in zip(pred_opp, actual_opp)])
                              
        # Calculate probability based on differences
        total_diff = pos_diff + opp_diff
        return np.exp(-total_diff / self.transition_noise) 