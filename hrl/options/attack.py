from typing import Dict, Any
import numpy as np
from hrl.options.base import BaseOption

class AttackOption(BaseOption):
    """Option for attacking and tagging opponents."""
    
    def __init__(self):
        super().__init__("attack")
        self.target_opponent = None
        self.attack_radius = 5.0
        self.last_opponent_position = None
        
    def initiate(self, state: Dict[str, Any]) -> bool:
        """Initiate if we see an opponent and we're not in danger."""
        agent_has_flag = state.get("agent_has_flag", False)
        agent_is_tagged = state.get("agent_is_tagged", False)
        opponent_positions = state.get("opponent_positions", [])
        
        if not agent_has_flag and not agent_is_tagged and opponent_positions:
            self.target_opponent = opponent_positions[0]
            return True
        return False
        
    def terminate(self, state: Dict[str, Any]) -> bool:
        """Terminate if we lose sight of the opponent or if we're tagged."""
        agent_is_tagged = state.get("agent_is_tagged", False)
        opponent_positions = state.get("opponent_positions", [])
        
        if agent_is_tagged or not opponent_positions:
            return True
            
        # Check if opponent is too far away
        if self.target_opponent is not None:
            agent_position = state.get("agent_position", None)
            if agent_position is not None:
                distance = np.linalg.norm(agent_position - self.target_opponent)
                if distance > self.attack_radius * 2:
                    return True
                    
        return False
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """Get action to move towards and tag the opponent."""
        agent_position = state.get("agent_position", None)
        if agent_position is None or self.target_opponent is None:
            return np.zeros(2)
            
        # Calculate direction to opponent
        direction = self.target_opponent - agent_position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance
            
        # Adjust speed based on distance
        speed = min(1.0, distance / self.attack_radius)
        
        return direction * speed
        
    def get_reward(self, state: Dict[str, Any], action: np.ndarray, next_state: Dict[str, Any]) -> float:
        """Get reward based on attack effectiveness."""
        reward = 0.0
        
        # Reward for getting closer to opponent
        agent_position = state.get("agent_position", None)
        if agent_position is not None and self.target_opponent is not None:
            current_dist = np.linalg.norm(agent_position - self.target_opponent)
            next_dist = np.linalg.norm(next_state.get("agent_position", agent_position) - self.target_opponent)
            reward += (current_dist - next_dist) * 5.0
            
        # Large reward for tagging opponent
        if next_state.get("opponent_tagged", False):
            reward += 50.0
            
        # Penalty for being tagged
        if next_state.get("agent_is_tagged", False):
            reward -= 30.0
            
        return reward
        
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """Update attack strategy based on opponent movement."""
        opponent_positions = state.get("opponent_positions", [])
        if opponent_positions:
            self.last_opponent_position = self.target_opponent
            self.target_opponent = opponent_positions[0]
            
            # Adjust attack radius based on opponent behavior
            if self.last_opponent_position is not None:
                opponent_speed = np.linalg.norm(self.target_opponent - self.last_opponent_position)
                self.attack_radius = max(3.0, min(7.0, 5.0 + opponent_speed))
                
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.target_opponent = None
        self.attack_radius = 5.0
        self.last_opponent_position = None 