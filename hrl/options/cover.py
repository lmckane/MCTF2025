import numpy as np
from typing import Dict, Any, Optional
from hrl.options.base import BaseOption

class CoverOption(BaseOption):
    """Option for providing cover and support for teammates."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cover option.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - cover_radius: Radius to search for cover opportunities (default: 25.0)
                - cover_speed: Speed multiplier when covering (default: 0.7)
                - min_teammate_distance: Minimum distance to maintain from covered teammate (default: 8.0)
                - max_teammate_distance: Maximum distance to maintain from covered teammate (default: 20.0)
                - threat_radius: Radius to consider opponents as threats (default: 30.0)
        """
        super().__init__("cover", config)
        self.cover_radius = self.config.get('cover_radius', 25.0)
        self.cover_speed = self.config.get('cover_speed', 0.7)
        self.min_teammate_distance = self.config.get('min_teammate_distance', 8.0)
        self.max_teammate_distance = self.config.get('max_teammate_distance', 20.0)
        self.threat_radius = self.config.get('threat_radius', 30.0)
        self.covered_teammate = None
        self.cover_position = None
        self.last_position = None
        self.steps_without_progress = 0
        
    def initiate(self, state: Dict[str, Any]) -> bool:
        """
        Initiate if we see a teammate that needs cover.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to initiate this option
        """
        opponent_positions = state.get("opponent_positions", [])
        teammate_positions = state.get("teammate_positions", [])
        agent_position = state.get("agent_position", None)
        agent_has_flag = state.get("agent_has_flag", False)
        
        if not opponent_positions or not teammate_positions or agent_position is None or agent_has_flag:
            return False
            
        agent_pos = np.array(agent_position)
        
        # Find teammate that needs cover
        for teammate_pos in teammate_positions:
            teammate_pos = np.array(teammate_pos)
            
            # Check if teammate has flag or is under threat
            teammate_has_flag = state.get("teammate_has_flag", False)
            is_under_threat = any(
                np.linalg.norm(np.array(opp_pos) - teammate_pos) < self.threat_radius
                for opp_pos in opponent_positions
            )
            
            if teammate_has_flag or is_under_threat:
                # Check if we can provide effective cover
                distance_to_teammate = np.linalg.norm(teammate_pos - agent_pos)
                if self.min_teammate_distance <= distance_to_teammate <= self.max_teammate_distance:
                    self.covered_teammate = teammate_pos
                    self.cover_position = self._calculate_cover_position(agent_pos, teammate_pos, opponent_positions)
                    self.last_position = agent_pos
                    self.steps_without_progress = 0
                    return True
                    
        return False
        
    def terminate(self, state: Dict[str, Any]) -> bool:
        """
        Terminate if cover is no longer needed or if we're too far from teammate.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to terminate this option
        """
        if self.covered_teammate is None or self.cover_position is None:
            return True
            
        agent_position = state.get("agent_position", None)
        if agent_position is None:
            return True
            
        # Check if we're stuck
        if self.steps_without_progress >= 50:
            return True
            
        # Check if teammate still needs cover
        teammate_positions = state.get("teammate_positions", [])
        if not any(np.array_equal(self.covered_teammate, pos) for pos in teammate_positions):
            return True
            
        # Check if teammate is too far
        distance = np.linalg.norm(self.covered_teammate - np.array(agent_position))
        if distance > self.max_teammate_distance:
            return True
            
        return False
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get action to maintain cover position.
        
        Args:
            state: Current environment state
            
        Returns:
            np.ndarray: Action to take [speed, heading]
        """
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is None or self.cover_position is None:
            return np.zeros(2)
            
        # Update cover position based on teammate and opponent movement
        opponent_positions = state.get("opponent_positions", [])
        if opponent_positions:
            self.cover_position = self._calculate_cover_position(
                agent_position,
                self.covered_teammate,
                opponent_positions
            )
            
        # Move to cover position
        direction = self.cover_position - agent_position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance
            
        # Check if we're making progress
        if self.last_position is not None:
            progress = np.linalg.norm(agent_position - self.last_position)
            if progress < 0.1:  # Threshold for considering movement
                self.steps_without_progress += 1
            else:
                self.steps_without_progress = 0
                
        self.last_position = agent_position
        
        # Calculate speed and heading
        speed = min(1.0, distance / 10.0) * self.cover_speed
        heading = np.arctan2(direction[1], direction[0]) * 180 / np.pi
        
        return np.array([speed, heading])
        
    def get_reward(self, state: Dict[str, Any], action: np.ndarray, next_state: Dict[str, Any]) -> float:
        """
        Get reward based on cover effectiveness.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            float: Reward value
        """
        reward = 0.0
        
        # Reward for maintaining cover position
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is not None and self.cover_position is not None:
            distance = np.linalg.norm(agent_position - self.cover_position)
            if distance < 2.0:
                reward += 1.0
                
        # Reward for being between teammate and threats
        opponent_positions = state.get("opponent_positions", [])
        if opponent_positions and self.covered_teammate is not None:
            for opp_pos in opponent_positions:
                opp_pos = np.array(opp_pos)
                if np.linalg.norm(opp_pos - self.covered_teammate) < self.threat_radius:
                    # Check if we're between opponent and teammate
                    if self._is_between(agent_position, opp_pos, self.covered_teammate):
                        reward += 2.0
                        
        # Large reward for preventing teammate from being tagged
        if not next_state.get("teammate_tagged", False):
            reward += 3.0
            
        # Penalty for being tagged
        if next_state.get("agent_is_tagged", False):
            reward -= 30.0
            
        # Penalty for being too far from teammate
        if self.covered_teammate is not None:
            distance = np.linalg.norm(self.covered_teammate - agent_position)
            if distance > self.max_teammate_distance:
                reward -= 5.0
                
        return reward
        
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """
        Update cover strategy based on effectiveness.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Adjust cover speed based on success
        if reward > 0:
            self.cover_speed = min(1.0, self.cover_speed + 0.01)
        else:
            self.cover_speed = max(0.5, self.cover_speed - 0.01)
            
    def _calculate_cover_position(self, agent_pos: np.ndarray, teammate_pos: np.ndarray, opponent_positions: list) -> np.ndarray:
        """Calculate optimal cover position."""
        if not opponent_positions:
            # If no opponents, maintain position relative to teammate
            direction = teammate_pos - agent_pos
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            return teammate_pos - direction * self.min_teammate_distance
            
        # Find the most threatening opponent
        most_threatening = max(
            opponent_positions,
            key=lambda pos: np.linalg.norm(np.array(pos) - teammate_pos)
        )
        most_threatening = np.array(most_threatening)
        
        # Calculate position between teammate and threat
        direction = most_threatening - teammate_pos
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            
        # Position ourselves between teammate and threat
        cover_distance = self.min_teammate_distance * 1.5
        return teammate_pos + direction * cover_distance
        
    def _is_between(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> bool:
        """Check if a point is between two other points."""
        # Calculate vectors
        line_vector = line_end - line_start
        point_vector = point - line_start
        
        # Check if point is in the right direction
        if np.dot(line_vector, point_vector) < 0:
            return False
            
        # Check if point is within the line segment
        line_length = np.linalg.norm(line_vector)
        point_projection = np.dot(point_vector, line_vector) / line_length
        
        return 0 <= point_projection <= line_length
        
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.covered_teammate = None
        self.cover_position = None
        self.last_position = None
        self.steps_without_progress = 0 