import numpy as np
from typing import Dict, Any, Optional
from hrl.options.base import BaseOption

class GuardFlagOption(BaseOption):
    """Option for guarding the team's flag."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the guard flag option.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - patrol_radius: Radius for patrolling around flag (default: 5.0)
                - num_patrol_points: Number of points in patrol path (default: 8)
                - intercept_distance: Distance at which to intercept opponents (default: 20.0)
                - guard_radius: Maximum distance to maintain from flag (default: 30.0)
                - patrol_speed: Speed multiplier for patrolling (default: 0.8)
        """
        super().__init__("guard_flag", config)
        self.flag_position = None
        self.patrol_points = []
        self.current_patrol_index = 0
        self.patrol_radius = self.config.get('patrol_radius', 5.0)
        self.num_patrol_points = self.config.get('num_patrol_points', 8)
        self.intercept_distance = self.config.get('intercept_distance', 20.0)
        self.guard_radius = self.config.get('guard_radius', 30.0)
        self.patrol_speed = self.config.get('patrol_speed', 0.8)
        self.target_opponent = None
        
    def initiate(self, state: Dict[str, Any]) -> bool:
        """
        Initiate if we have our flag and no opponent is nearby.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to initiate this option
        """
        agent_has_flag = state.get("agent_has_flag", False)
        flag_position = state.get("team_flag_position", None)
        opponent_positions = state.get("opponent_positions", [])
        agent_position = state.get("agent_position", None)
        
        if not agent_has_flag and flag_position is not None and agent_position is not None:
            # Check if any opponents are too close
            for opp_pos in opponent_positions:
                if np.linalg.norm(np.array(opp_pos) - np.array(flag_position)) < self.intercept_distance:
                    return False
                    
            self.flag_position = np.array(flag_position)
            self._generate_patrol_points()
            return True
        return False
        
    def terminate(self, state: Dict[str, Any]) -> bool:
        """
        Terminate if we lose our flag or if an opponent is too close.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to terminate this option
        """
        agent_has_flag = state.get("agent_has_flag", False)
        opponent_positions = state.get("opponent_positions", [])
        agent_position = state.get("agent_position", None)
        
        if agent_has_flag:
            return True
            
        if agent_position is not None and self.flag_position is not None:
            # Check if any opponent is too close to flag
            for opp_pos in opponent_positions:
                if np.linalg.norm(np.array(opp_pos) - self.flag_position) < self.intercept_distance:
                    return True
                    
        return False
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get action to patrol around the flag or intercept opponents.
        
        Args:
            state: Current environment state
            
        Returns:
            np.ndarray: Action to take [speed, heading]
        """
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is None or self.flag_position is None:
            return np.zeros(2)
            
        # Check for nearby opponents
        opponent_positions = state.get("opponent_positions", [])
        for opp_pos in opponent_positions:
            opp_pos = np.array(opp_pos)
            if np.linalg.norm(opp_pos - self.flag_position) < self.intercept_distance:
                self.target_opponent = opp_pos
                break
                
        if self.target_opponent is not None:
            # Intercept opponent
            direction = self.target_opponent - agent_position
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                direction = direction / distance
                
            # Calculate speed and heading
            speed = min(1.0, distance / 10.0)
            heading = np.arctan2(direction[1], direction[0]) * 180 / np.pi
            
            return np.array([speed, heading])
            
        else:
            # Patrol around flag
            if not self.patrol_points:
                self._generate_patrol_points()
                
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
                
            # Calculate speed and heading
            speed = min(1.0, distance / 10.0) * self.patrol_speed
            heading = np.arctan2(direction[1], direction[0]) * 180 / np.pi
            
            return np.array([speed, heading])
            
    def get_reward(self, state: Dict[str, Any], action: np.ndarray, next_state: Dict[str, Any]) -> float:
        """
        Get reward based on patrol effectiveness and flag protection.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            float: Reward value
        """
        reward = 0.0
        
        # Reward for staying near flag
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is not None and self.flag_position is not None:
            distance_to_flag = np.linalg.norm(agent_position - self.flag_position)
            if distance_to_flag <= self.patrol_radius:
                reward += 1.0
                
        # Penalty for being too far from flag
        if distance_to_flag > self.guard_radius:
            reward -= 2.0
            
        # Reward for intercepting opponents
        opponent_positions = state.get("opponent_positions", [])
        for opp_pos in opponent_positions:
            opp_pos = np.array(opp_pos)
            if np.linalg.norm(opp_pos - agent_position) < self.intercept_distance:
                reward += 5.0
                
        # Large reward for tagging opponent
        if next_state.get("opponent_tagged", False):
            reward += 40.0
            
        # Penalty for being tagged
        if next_state.get("agent_is_tagged", False):
            reward -= 20.0
            
        return reward
        
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """
        Update patrol strategy based on opponent positions.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        opponent_positions = state.get("opponent_positions", [])
        if opponent_positions:
            # Adjust patrol radius based on opponent proximity
            min_opp_dist = min(
                np.linalg.norm(np.array(opp_pos) - self.flag_position)
                for opp_pos in opponent_positions
            )
            self.patrol_radius = max(3.0, min(7.0, min_opp_dist * 0.8))
            self._generate_patrol_points()
            
        # Adjust patrol speed based on success
        if reward > 0:
            self.patrol_speed = min(1.0, self.patrol_speed + 0.01)
        else:
            self.patrol_speed = max(0.5, self.patrol_speed - 0.01)
            
    def _generate_patrol_points(self):
        """Generate patrol points around the flag."""
        if self.flag_position is None:
            return
            
        angles = np.linspace(0, 2*np.pi, self.num_patrol_points, endpoint=False)
        self.patrol_points = [
            self.flag_position + self.patrol_radius * np.array([np.cos(a), np.sin(a)])
            for a in angles
        ]
        
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.flag_position = None
        self.patrol_points = []
        self.current_patrol_index = 0
        self.target_opponent = None 