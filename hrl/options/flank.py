import numpy as np
from typing import Dict, Any, Optional
from hrl.options.base import BaseOption

class FlankOption(BaseOption):
    """Option for flanking opponents and creating tactical advantages."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the flank option.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - flank_radius: Radius to search for flanking opportunities (default: 25.0)
                - flank_speed: Speed multiplier when flanking (default: 0.9)
                - min_opponent_distance: Minimum distance to maintain from opponents (default: 8.0)
                - max_teammate_distance: Maximum distance to maintain from teammates (default: 15.0)
        """
        super().__init__("flank", config)
        self.flank_radius = self.config.get('flank_radius', 25.0)
        self.flank_speed = self.config.get('flank_speed', 0.9)
        self.min_opponent_distance = self.config.get('min_opponent_distance', 8.0)
        self.max_teammate_distance = self.config.get('max_teammate_distance', 15.0)
        self.target_opponent = None
        self.flank_position = None
        self.last_position = None
        self.steps_without_progress = 0
        
    def initiate(self, state: Dict[str, Any]) -> bool:
        """
        Initiate if we see an opportunity to flank an opponent.
        
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
        
        # Find opponent that can be flanked
        for opp_pos in opponent_positions:
            opp_pos = np.array(opp_pos)
            
            # Check if opponent is engaged with a teammate
            for teammate_pos in teammate_positions:
                teammate_pos = np.array(teammate_pos)
                if np.linalg.norm(opp_pos - teammate_pos) < self.flank_radius:
                    # Check if we can flank from a different angle
                    angle = self._calculate_flank_angle(agent_pos, opp_pos, teammate_pos)
                    if angle > 45.0:  # Good flanking angle
                        self.target_opponent = opp_pos
                        self.flank_position = self._calculate_flank_position(agent_pos, opp_pos, teammate_pos)
                        self.last_position = agent_pos
                        self.steps_without_progress = 0
                        return True
                        
        return False
        
    def terminate(self, state: Dict[str, Any]) -> bool:
        """
        Terminate if flanking opportunity is lost or if we're too far from target.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to terminate this option
        """
        if self.target_opponent is None or self.flank_position is None:
            return True
            
        agent_position = state.get("agent_position", None)
        if agent_position is None:
            return True
            
        # Check if we're stuck
        if self.steps_without_progress >= 50:
            return True
            
        # Check if opponent is still in flanking position
        opponent_positions = state.get("opponent_positions", [])
        teammate_positions = state.get("teammate_positions", [])
        
        if not any(
            np.linalg.norm(np.array(opp_pos) - np.array(teammate_pos)) < self.flank_radius
            for opp_pos in opponent_positions
            for teammate_pos in teammate_positions
        ):
            return True
            
        return False
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get action to execute flanking maneuver.
        
        Args:
            state: Current environment state
            
        Returns:
            np.ndarray: Action to take [speed, heading]
        """
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is None or self.flank_position is None:
            return np.zeros(2)
            
        # Update flank position based on opponent movement
        opponent_positions = state.get("opponent_positions", [])
        teammate_positions = state.get("teammate_positions", [])
        
        if opponent_positions and teammate_positions:
            # Find the engaged opponent
            for opp_pos in opponent_positions:
                opp_pos = np.array(opp_pos)
                for teammate_pos in teammate_positions:
                    teammate_pos = np.array(teammate_pos)
                    if np.linalg.norm(opp_pos - teammate_pos) < self.flank_radius:
                        self.target_opponent = opp_pos
                        self.flank_position = self._calculate_flank_position(agent_position, opp_pos, teammate_pos)
                        break
                        
        # Move to flank position
        direction = self.flank_position - agent_position
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
        speed = min(1.0, distance / 10.0) * self.flank_speed
        heading = np.arctan2(direction[1], direction[0]) * 180 / np.pi
        
        return np.array([speed, heading])
        
    def get_reward(self, state: Dict[str, Any], action: np.ndarray, next_state: Dict[str, Any]) -> float:
        """
        Get reward based on flanking effectiveness.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            float: Reward value
        """
        reward = 0.0
        
        # Reward for maintaining flank position
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is not None and self.flank_position is not None:
            distance = np.linalg.norm(agent_position - self.flank_position)
            if distance < 2.0:
                reward += 1.0
                
        # Reward for good flanking angle
        opponent_positions = state.get("opponent_positions", [])
        teammate_positions = state.get("teammate_positions", [])
        
        if opponent_positions and teammate_positions:
            for opp_pos in opponent_positions:
                opp_pos = np.array(opp_pos)
                for teammate_pos in teammate_positions:
                    teammate_pos = np.array(teammate_pos)
                    angle = self._calculate_flank_angle(agent_position, opp_pos, teammate_pos)
                    if angle > 60.0:  # Excellent flanking angle
                        reward += 5.0
                        
        # Large reward for tagging opponent from flank
        if next_state.get("opponent_tagged", False):
            reward += 60.0
            
        # Penalty for being tagged
        if next_state.get("agent_is_tagged", False):
            reward -= 30.0
            
        # Penalty for being too close to opponent
        for opp_pos in opponent_positions:
            opp_pos = np.array(opp_pos)
            if np.linalg.norm(opp_pos - agent_position) < self.min_opponent_distance:
                reward -= 10.0
                
        return reward
        
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """
        Update flanking strategy based on opponent behavior.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Adjust flank speed based on success
        if reward > 0:
            self.flank_speed = min(1.1, self.flank_speed + 0.01)
        else:
            self.flank_speed = max(0.7, self.flank_speed - 0.01)
            
    def _calculate_flank_angle(self, agent_pos: np.ndarray, opp_pos: np.ndarray, teammate_pos: np.ndarray) -> float:
        """Calculate the angle between agent, opponent, and teammate."""
        v1 = opp_pos - agent_pos
        v2 = opp_pos - teammate_pos
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0:
            return 0.0
        angle = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
        return np.degrees(angle)
        
    def _calculate_flank_position(self, agent_pos: np.ndarray, opp_pos: np.ndarray, teammate_pos: np.ndarray) -> np.ndarray:
        """Calculate optimal flanking position."""
        # Calculate perpendicular direction from opponent to teammate
        direction = teammate_pos - opp_pos
        perpendicular = np.array([-direction[1], direction[0]])
        if np.linalg.norm(perpendicular) > 0:
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
            
        # Position ourselves at a good flanking distance
        flank_distance = self.min_opponent_distance * 1.5
        return opp_pos + perpendicular * flank_distance
        
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.target_opponent = None
        self.flank_position = None
        self.last_position = None
        self.steps_without_progress = 0 