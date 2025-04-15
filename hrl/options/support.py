from typing import Dict, Any
import numpy as np
from hrl.options.base import BaseOption

class SupportOption(BaseOption):
    """Option for supporting teammates in various tasks."""
    
    def __init__(self):
        super().__init__("support")
        self.support_target = None
        self.support_type = None  # 'escort', 'cover', or 'assist'
        self.support_radius = 6.0
        self.last_target_position = None
        
    def initiate(self, state: Dict[str, Any]) -> bool:
        """Initiate if a teammate needs support."""
        agent_has_flag = state.get("agent_has_flag", False)
        agent_is_tagged = state.get("agent_is_tagged", False)
        teammate_positions = state.get("teammate_positions", [])
        teammate_states = state.get("teammate_states", [])
        
        if not agent_has_flag and not agent_is_tagged and teammate_positions:
            # Check each teammate's state
            for i, (teammate_pos, teammate_state) in enumerate(zip(teammate_positions, teammate_states)):
                # Support flag carrier
                if teammate_state.get("has_flag", False):
                    self.support_target = teammate_pos
                    self.support_type = "escort"
                    return True
                    
                # Support teammate under attack
                if teammate_state.get("is_under_attack", False):
                    self.support_target = teammate_pos
                    self.support_type = "cover"
                    return True
                    
                # Support teammate near opponent flag
                if teammate_state.get("near_opponent_flag", False):
                    self.support_target = teammate_pos
                    self.support_type = "assist"
                    return True
                    
        return False
        
    def terminate(self, state: Dict[str, Any]) -> bool:
        """Terminate if support is no longer needed or if we're tagged."""
        agent_is_tagged = state.get("agent_is_tagged", False)
        teammate_states = state.get("teammate_states", [])
        
        if agent_is_tagged:
            return True
            
        # Check if support is still needed
        if self.support_type == "escort":
            for teammate_state in teammate_states:
                if not teammate_state.get("has_flag", False):
                    return True
        elif self.support_type == "cover":
            for teammate_state in teammate_states:
                if not teammate_state.get("is_under_attack", False):
                    return True
        elif self.support_type == "assist":
            for teammate_state in teammate_states:
                if not teammate_state.get("near_opponent_flag", False):
                    return True
                    
        return False
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """Get action to provide support based on support type."""
        agent_position = state.get("agent_position", None)
        if agent_position is None or self.support_target is None:
            return np.zeros(2)
            
        if self.support_type == "escort":
            # Stay slightly behind the flag carrier
            direction = self.support_target - agent_position
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                direction = direction / distance
                target_position = self.support_target - direction * (self.support_radius / 2)
                move_direction = target_position - agent_position
                move_distance = np.linalg.norm(move_direction)
                
                if move_distance > 0:
                    return move_direction / move_distance
                    
        elif self.support_type == "cover":
            # Circle around the teammate under attack
            direction = self.support_target - agent_position
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                direction = direction / distance
                # Move perpendicular to the direction to teammate
                perpendicular = np.array([-direction[1], direction[0]])
                target_position = self.support_target + perpendicular * self.support_radius
                move_direction = target_position - agent_position
                move_distance = np.linalg.norm(move_direction)
                
                if move_distance > 0:
                    return move_direction / move_distance
                    
        elif self.support_type == "assist":
            # Move to assist in capturing the flag
            direction = self.support_target - agent_position
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                direction = direction / distance
                return direction
                
        return np.zeros(2)
        
    def get_reward(self, state: Dict[str, Any], action: np.ndarray, next_state: Dict[str, Any]) -> float:
        """Get reward based on support effectiveness."""
        reward = 0.0
        
        # Reward for maintaining support position
        agent_position = state.get("agent_position", None)
        if agent_position is not None and self.support_target is not None:
            distance_to_target = np.linalg.norm(agent_position - self.support_target)
            if distance_to_target <= self.support_radius:
                reward += 1.0
                
        # Reward for successful support
        teammate_states = next_state.get("teammate_states", [])
        for teammate_state in teammate_states:
            if self.support_type == "escort" and teammate_state.get("flag_captured", False):
                reward += 30.0
            elif self.support_type == "cover" and not teammate_state.get("is_tagged", False):
                reward += 20.0
            elif self.support_type == "assist" and teammate_state.get("flag_captured", False):
                reward += 25.0
                
        # Penalty for being tagged
        if next_state.get("agent_is_tagged", False):
            reward -= 15.0
            
        return reward
        
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """Update support strategy based on teammate needs."""
        teammate_positions = state.get("teammate_positions", [])
        if teammate_positions:
            self.last_target_position = self.support_target
            self.support_target = teammate_positions[0]
            
            # Adjust support radius based on situation
            if self.support_type == "escort":
                self.support_radius = 4.0
            elif self.support_type == "cover":
                self.support_radius = 6.0
            elif self.support_type == "assist":
                self.support_radius = 8.0
                
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.support_target = None
        self.support_type = None
        self.support_radius = 6.0
        self.last_target_position = None 