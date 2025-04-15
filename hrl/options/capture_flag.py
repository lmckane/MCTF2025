from typing import Dict, Any
import numpy as np
from hrl.options.base import BaseOption

class CaptureFlagOption(BaseOption):
    """Option for capturing the opponent's flag."""
    
    def __init__(self):
        super().__init__("capture_flag")
        self.target_flag = None
        self.path_to_flag = []
        
    def initiate(self, state: Dict[str, Any]) -> bool:
        """Initiate if we don't have our flag and the opponent's flag is reachable."""
        agent_has_flag = state.get("agent_has_flag", False)
        opponent_flag_position = state.get("opponent_flag_position", None)
        agent_position = state.get("agent_position", None)
        
        if not agent_has_flag and opponent_flag_position is not None and agent_position is not None:
            self.target_flag = opponent_flag_position
            return True
        return False
        
    def terminate(self, state: Dict[str, Any]) -> bool:
        """Terminate if we have the flag or if we're tagged."""
        agent_has_flag = state.get("agent_has_flag", False)
        agent_is_tagged = state.get("agent_is_tagged", False)
        return agent_has_flag or agent_is_tagged
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """Get action to move towards the opponent's flag."""
        agent_position = state.get("agent_position", None)
        if agent_position is None or self.target_flag is None:
            return np.zeros(2)  # No movement if we don't have position information
            
        # Calculate direction to flag
        direction = self.target_flag - agent_position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance  # Normalize
            
        # Return action (normalized direction vector)
        return direction
        
    def get_reward(self, state: Dict[str, Any], action: np.ndarray, next_state: Dict[str, Any]) -> float:
        """Get reward based on progress towards flag and successful capture."""
        reward = 0.0
        
        # Reward for getting closer to flag
        current_dist = np.linalg.norm(state.get("agent_position", np.zeros(2)) - self.target_flag)
        next_dist = np.linalg.norm(next_state.get("agent_position", np.zeros(2)) - self.target_flag)
        reward += (current_dist - next_dist) * 10.0  # Reward for moving closer
        
        # Large reward for capturing flag
        if next_state.get("agent_has_flag", False):
            reward += 100.0
            
        # Penalty for being tagged
        if next_state.get("agent_is_tagged", False):
            reward -= 50.0
            
        return reward
        
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """Update internal state and path planning."""
        # Update path planning if needed
        if self.target_flag is not None:
            agent_position = state.get("agent_position", None)
            if agent_position is not None:
                # Simple path planning: move directly towards flag
                direction = self.target_flag - agent_position
                distance = np.linalg.norm(direction)
                if distance > 0:
                    self.path_to_flag = [agent_position + direction * (i/10) for i in range(1, 11)]
                    
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.target_flag = None
        self.path_to_flag = [] 