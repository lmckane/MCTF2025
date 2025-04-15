import numpy as np
from typing import Dict, Any
from hrl.options.base import BaseOption

class GuardFlagOption(BaseOption):
    """Option for guarding the team's flag."""
    
    def __init__(self):
        super().__init__("guard_flag")
        self.flag_position = None
        self.patrol_points = []
        self.current_patrol_index = 0
        self.patrol_radius = 5.0
        
    def initiate(self, state: Dict[str, Any]) -> bool:
        """Initiate if we have our flag and no opponent is nearby."""
        agent_has_flag = state.get("agent_has_flag", False)
        flag_position = state.get("team_flag_position", None)
        opponent_positions = state.get("opponent_positions", [])
        agent_position = state.get("agent_position", None)
        
        if not agent_has_flag and flag_position is not None and agent_position is not None:
            self.flag_position = flag_position
            self._generate_patrol_points()
            return True
        return False
        
    def terminate(self, state: Dict[str, Any]) -> bool:
        """Terminate if we lose our flag or if an opponent is too close."""
        agent_has_flag = state.get("agent_has_flag", False)
        opponent_positions = state.get("opponent_positions", [])
        agent_position = state.get("agent_position", None)
        
        if agent_has_flag:
            return True
            
        if agent_position is not None and self.flag_position is not None:
            for opp_pos in opponent_positions:
                if np.linalg.norm(opp_pos - self.flag_position) < self.patrol_radius:
                    return True
                    
        return False
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """Get action to patrol around the flag."""
        agent_position = state.get("agent_position", None)
        if agent_position is None or not self.patrol_points:
            return np.zeros(2)
            
        # Move to next patrol point
        target = self.patrol_points[self.current_patrol_index]
        direction = target - agent_position
        distance = np.linalg.norm(direction)
        
        if distance < 0.5:  # Reached current patrol point
            self.current_patrol_index = (self.current_patrol_index + 1) % len(self.patrol_points)
            target = self.patrol_points[self.current_patrol_index]
            direction = target - agent_position
            distance = np.linalg.norm(direction)
            
        if distance > 0:
            direction = direction / distance
            
        return direction
        
    def get_reward(self, state: Dict[str, Any], action: np.ndarray, next_state: Dict[str, Any]) -> float:
        """Get reward based on patrol effectiveness and flag protection."""
        reward = 0.0
        
        # Reward for staying near flag
        agent_position = state.get("agent_position", None)
        if agent_position is not None and self.flag_position is not None:
            distance_to_flag = np.linalg.norm(agent_position - self.flag_position)
            if distance_to_flag <= self.patrol_radius:
                reward += 1.0
                
        # Penalty for being too far from flag
        if distance_to_flag > self.patrol_radius * 1.5:
            reward -= 2.0
            
        # Reward for intercepting opponents
        opponent_positions = state.get("opponent_positions", [])
        for opp_pos in opponent_positions:
            if np.linalg.norm(opp_pos - agent_position) < self.patrol_radius:
                reward += 5.0
                
        return reward
        
    def _generate_patrol_points(self):
        """Generate patrol points around the flag."""
        if self.flag_position is None:
            return
            
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        self.patrol_points = [
            self.flag_position + self.patrol_radius * np.array([np.cos(a), np.sin(a)])
            for a in angles
        ]
        
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """Update patrol strategy based on opponent positions."""
        opponent_positions = state.get("opponent_positions", [])
        if opponent_positions:
            # Adjust patrol radius based on opponent proximity
            min_opp_dist = min(
                np.linalg.norm(opp_pos - self.flag_position)
                for opp_pos in opponent_positions
            )
            self.patrol_radius = max(3.0, min(7.0, min_opp_dist * 0.8))
            self._generate_patrol_points()
            
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.flag_position = None
        self.patrol_points = []
        self.current_patrol_index = 0
        self.patrol_radius = 5.0 