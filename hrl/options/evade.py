import numpy as np
from typing import Dict, Any, Optional
from hrl.options.base import BaseOption

class EvadeOption(BaseOption):
    """Option for evading the nearest opponent."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evade option.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - evade_radius: Radius to start evading (default: 15.0)
                - evade_speed: Speed multiplier when evading (default: 1.0)
                - min_safe_distance: Minimum safe distance from opponents (default: 20.0)
                - max_evade_angle: Maximum angle to turn when evading (default: 45.0)
        """
        super().__init__("evade", config)
        self.evade_radius = self.config.get('evade_radius', 15.0)
        self.evade_speed = self.config.get('evade_speed', 1.0)
        self.min_safe_distance = self.config.get('min_safe_distance', 20.0)
        self.max_evade_angle = self.config.get('max_evade_angle', 45.0)
        self.nearest_opponent = None
        self.evade_direction = None
        self.last_position = None
        self.steps_without_progress = 0
        
    def initiate(self, state: Dict[str, Any]) -> bool:
        """
        Initiate if an opponent is too close.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to initiate this option
        """
        opponent_positions = state.get("opponent_positions", [])
        agent_position = state.get("agent_position", None)
        
        if not opponent_positions or agent_position is None:
            return False
            
        agent_pos = np.array(agent_position)
        
        # Find nearest opponent
        nearest_opp = min(
            opponent_positions,
            key=lambda pos: np.linalg.norm(np.array(pos) - agent_pos)
        )
        nearest_opp = np.array(nearest_opp)
        distance = np.linalg.norm(nearest_opp - agent_pos)
        
        if distance < self.evade_radius:
            self.nearest_opponent = nearest_opp
            self.evade_direction = self._calculate_evade_direction(agent_pos, nearest_opp)
            self.last_position = agent_pos
            self.steps_without_progress = 0
            return True
            
        return False
        
    def terminate(self, state: Dict[str, Any]) -> bool:
        """
        Terminate if we're safe or stuck.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to terminate this option
        """
        if self.nearest_opponent is None or self.evade_direction is None:
            return True
            
        agent_position = state.get("agent_position", None)
        if agent_position is None:
            return True
            
        # Check if we're stuck
        if self.steps_without_progress >= 50:
            return True
            
        # Check if we're safe
        distance = np.linalg.norm(self.nearest_opponent - np.array(agent_position))
        if distance >= self.min_safe_distance:
            return True
            
        return False
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get action to evade the nearest opponent.
        
        Args:
            state: Current environment state
            
        Returns:
            np.ndarray: Action to take [speed, heading]
        """
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is None or self.evade_direction is None:
            return np.zeros(2)
            
        # Update evade direction based on opponent movement
        opponent_positions = state.get("opponent_positions", [])
        if opponent_positions:
            nearest_opp = min(
                opponent_positions,
                key=lambda pos: np.linalg.norm(np.array(pos) - agent_position)
            )
            self.nearest_opponent = np.array(nearest_opp)
            self.evade_direction = self._calculate_evade_direction(agent_position, nearest_opp)
            
        # Check if we're making progress
        if self.last_position is not None:
            progress = np.linalg.norm(agent_position - self.last_position)
            if progress < 0.1:  # Threshold for considering movement
                self.steps_without_progress += 1
            else:
                self.steps_without_progress = 0
                
        self.last_position = agent_position
        
        # Calculate speed and heading
        speed = self.evade_speed
        heading = np.arctan2(self.evade_direction[1], self.evade_direction[0]) * 180 / np.pi
        
        return np.array([speed, heading])
        
    def get_reward(self, state: Dict[str, Any], action: np.ndarray, next_state: Dict[str, Any]) -> float:
        """
        Get reward based on evasion effectiveness.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            float: Reward value
        """
        reward = 0.0
        
        # Reward for increasing distance from opponent
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is not None and self.nearest_opponent is not None:
            current_distance = np.linalg.norm(self.nearest_opponent - agent_position)
            if current_distance > self.evade_radius:
                reward += 1.0
                
        # Large reward for reaching safe distance
        if current_distance >= self.min_safe_distance:
            reward += 5.0
            
        # Penalty for being tagged
        if next_state.get("agent_is_tagged", False):
            reward -= 30.0
            
        # Penalty for getting closer to opponent
        next_agent_pos = np.array(next_state.get("agent_position", [0, 0]))
        next_distance = np.linalg.norm(self.nearest_opponent - next_agent_pos)
        if next_distance < current_distance:
            reward -= 2.0
            
        return reward
        
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """
        Update evasion strategy based on effectiveness.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Adjust evade speed based on success
        if reward > 0:
            self.evade_speed = min(1.2, self.evade_speed + 0.01)
        else:
            self.evade_speed = max(0.8, self.evade_speed - 0.01)
            
    def _calculate_evade_direction(self, agent_pos: np.ndarray, opponent_pos: np.ndarray) -> np.ndarray:
        """Calculate direction to evade opponent."""
        # Calculate direction away from opponent
        direction = agent_pos - opponent_pos
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            
        # Add some randomness to avoid predictable patterns
        angle = np.random.uniform(-self.max_evade_angle, self.max_evade_angle)
        angle_rad = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        return rotation_matrix @ direction
        
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.nearest_opponent = None
        self.evade_direction = None
        self.last_position = None
        self.steps_without_progress = 0 