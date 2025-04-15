from typing import Dict, Any
import numpy as np
from hrl.options.base import BaseOption

class DefendOption(BaseOption):
    """Option for defending against opponents and protecting teammates."""
    
    def __init__(self):
        super().__init__("defend")
        self.target_opponent = None
        self.defense_position = None
        self.defense_radius = 8.0
        self.last_opponent_position = None
        
    def initiate(self, state: Dict[str, Any]) -> bool:
        """Initiate if we see an opponent near our flag or teammates."""
        agent_has_flag = state.get("agent_has_flag", False)
        agent_is_tagged = state.get("agent_is_tagged", False)
        opponent_positions = state.get("opponent_positions", [])
        team_flag_position = state.get("team_flag_position", None)
        teammate_positions = state.get("teammate_positions", [])
        
        if not agent_has_flag and not agent_is_tagged and opponent_positions:
            # Check if opponent is near flag or teammates
            for opp_pos in opponent_positions:
                if team_flag_position is not None:
                    flag_dist = np.linalg.norm(opp_pos - team_flag_position)
                    if flag_dist < self.defense_radius:
                        self.target_opponent = opp_pos
                        self.defense_position = team_flag_position
                        return True
                        
                for teammate_pos in teammate_positions:
                    teammate_dist = np.linalg.norm(opp_pos - teammate_pos)
                    if teammate_dist < self.defense_radius:
                        self.target_opponent = opp_pos
                        self.defense_position = teammate_pos
                        return True
                        
        return False
        
    def terminate(self, state: Dict[str, Any]) -> bool:
        """Terminate if we lose sight of the opponent or if we're tagged."""
        agent_is_tagged = state.get("agent_is_tagged", False)
        opponent_positions = state.get("opponent_positions", [])
        
        if agent_is_tagged or not opponent_positions:
            return True
            
        # Check if opponent is too far from defense position
        if self.target_opponent is not None and self.defense_position is not None:
            distance = np.linalg.norm(self.target_opponent - self.defense_position)
            if distance > self.defense_radius * 2:
                return True
                
        return False
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """Get action to intercept the opponent while maintaining defense position."""
        agent_position = state.get("agent_position", None)
        if agent_position is None or self.target_opponent is None or self.defense_position is None:
            return np.zeros(2)
            
        # Calculate intercept point between opponent and defense position
        opp_to_defense = self.defense_position - self.target_opponent
        opp_to_defense_dist = np.linalg.norm(opp_to_defense)
        
        if opp_to_defense_dist > 0:
            opp_to_defense = opp_to_defense / opp_to_defense_dist
            intercept_point = self.target_opponent + opp_to_defense * min(opp_to_defense_dist/2, self.defense_radius)
            
            # Move towards intercept point
            direction = intercept_point - agent_position
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                direction = direction / distance
                
            # Adjust speed based on distance
            speed = min(1.0, distance / self.defense_radius)
            
            return direction * speed
            
        return np.zeros(2)
        
    def get_reward(self, state: Dict[str, Any], action: np.ndarray, next_state: Dict[str, Any]) -> float:
        """Get reward based on defense effectiveness."""
        reward = 0.0
        
        # Reward for maintaining defense position
        agent_position = state.get("agent_position", None)
        if agent_position is not None and self.defense_position is not None:
            distance_to_defense = np.linalg.norm(agent_position - self.defense_position)
            if distance_to_defense <= self.defense_radius:
                reward += 1.0
                
        # Reward for intercepting opponent
        if self.target_opponent is not None:
            current_dist = np.linalg.norm(agent_position - self.target_opponent)
            next_dist = np.linalg.norm(next_state.get("agent_position", agent_position) - self.target_opponent)
            reward += (current_dist - next_dist) * 3.0
            
        # Large reward for tagging opponent
        if next_state.get("opponent_tagged", False):
            reward += 40.0
            
        # Penalty for being tagged
        if next_state.get("agent_is_tagged", False):
            reward -= 20.0
            
        return reward
        
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """Update defense strategy based on opponent movement."""
        opponent_positions = state.get("opponent_positions", [])
        if opponent_positions:
            self.last_opponent_position = self.target_opponent
            self.target_opponent = opponent_positions[0]
            
            # Adjust defense radius based on opponent behavior
            if self.last_opponent_position is not None:
                opponent_speed = np.linalg.norm(self.target_opponent - self.last_opponent_position)
                self.defense_radius = max(5.0, min(10.0, 8.0 + opponent_speed))
                
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.target_opponent = None
        self.defense_position = None
        self.defense_radius = 8.0
        self.last_opponent_position = None 