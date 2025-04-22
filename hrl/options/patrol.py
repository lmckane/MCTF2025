import numpy as np
from typing import Dict, Any, Optional
from hrl.options.base import BaseOption

class PatrolOption(BaseOption):
    """Option for patrolling and scouting the environment."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the patrol option.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - patrol_radius: Radius for patrolling (default: 20.0)
                - num_patrol_points: Number of points in patrol path (default: 8)
                - patrol_speed: Speed multiplier when patrolling (default: 0.7)
                - min_distance: Minimum distance to maintain from obstacles (default: 5.0)
        """
        super().__init__("patrol", config)
        self.patrol_radius = self.config.get('patrol_radius', 20.0)
        self.num_patrol_points = self.config.get('num_patrol_points', 8)
        self.patrol_speed = self.config.get('patrol_speed', 0.7)
        self.min_distance = self.config.get('min_distance', 5.0)
        self.patrol_points = []
        self.current_patrol_index = 0
        self.patrol_center = None
        
    def initiate(self, state: Dict[str, Any]) -> bool:
        """
        Initiate if we're not engaged in other tasks.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to initiate this option
        """
        agent_has_flag = state.get("agent_has_flag", False)
        agent_is_tagged = state.get("agent_is_tagged", False)
        agent_position = state.get("agent_position", None)
        
        if not agent_has_flag and not agent_is_tagged and agent_position is not None:
            self.patrol_center = np.array(agent_position)
            self._generate_patrol_points()
            return True
        return False
        
    def terminate(self, state: Dict[str, Any]) -> bool:
        """
        Terminate if we get the flag, get tagged, or see an opponent.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to terminate this option
        """
        agent_has_flag = state.get("agent_has_flag", False)
        agent_is_tagged = state.get("agent_is_tagged", False)
        opponent_positions = state.get("opponent_positions", [])
        agent_position = state.get("agent_position", None)
        
        if agent_has_flag or agent_is_tagged or not opponent_positions:
            return True
            
        # Check if we see an opponent
        if agent_position is not None:
            for opp_pos in opponent_positions:
                if np.linalg.norm(np.array(opp_pos) - np.array(agent_position)) < self.patrol_radius:
                    return True
                    
        return False
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get action to patrol the area.
        
        Args:
            state: Current environment state
            
        Returns:
            np.ndarray: Action to take [speed, heading]
        """
        agent_position = np.array(state.get("agent_position", [0, 0]))
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
            
        # Calculate speed and heading
        speed = min(1.0, distance / 10.0) * self.patrol_speed
        heading = np.arctan2(direction[1], direction[0]) * 180 / np.pi
        
        return np.array([speed, heading])
        
    def get_reward(self, state: Dict[str, Any], action: np.ndarray, next_state: Dict[str, Any]) -> float:
        """
        Get reward based on patrol effectiveness.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            float: Reward value
        """
        reward = 0.0
        
        # Reward for maintaining patrol path
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is not None and self.patrol_points:
            target = self.patrol_points[self.current_patrol_index]
            distance = np.linalg.norm(agent_position - target)
            if distance < 1.0:
                reward += 1.0
                
        # Reward for exploring new areas
        if next_state.get("new_area_explored", False):
            reward += 5.0
            
        # Penalty for being too close to obstacles
        obstacle_positions = state.get("obstacle_positions", [])
        for obs_pos in obstacle_positions:
            obs_pos = np.array(obs_pos)
            if np.linalg.norm(obs_pos - agent_position) < self.min_distance:
                reward -= 2.0
                
        return reward
        
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """
        Update patrol strategy based on environment.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Update patrol center if needed
        agent_position = state.get("agent_position", None)
        if agent_position is not None:
            self.patrol_center = np.array(agent_position)
            self._generate_patrol_points()
            
        # Adjust patrol speed based on success
        if reward > 0:
            self.patrol_speed = min(1.0, self.patrol_speed + 0.01)
        else:
            self.patrol_speed = max(0.5, self.patrol_speed - 0.01)
            
    def _generate_patrol_points(self):
        """Generate patrol points around the center."""
        if self.patrol_center is None:
            return
            
        angles = np.linspace(0, 2*np.pi, self.num_patrol_points, endpoint=False)
        self.patrol_points = [
            self.patrol_center + self.patrol_radius * np.array([np.cos(a), np.sin(a)])
            for a in angles
        ]
        
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.patrol_points = []
        self.current_patrol_index = 0
        self.patrol_center = None 