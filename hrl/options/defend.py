from typing import Dict, Any, Optional
import numpy as np
from hrl.options.base import BaseOption

class DefendOption(BaseOption):
    """Option for defending against opponents and protecting territory."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the defend option.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - defend_radius: Radius to defend around (default: 15.0)
                - intercept_distance: Distance at which to intercept opponents (default: 25.0)
                - defend_speed: Speed multiplier when defending (default: 0.8)
                - min_opponent_distance: Minimum distance to maintain from opponents (default: 5.0)
        """
        super().__init__("defend", config)
        self.defend_radius = self.config.get('defend_radius', 15.0)
        self.intercept_distance = self.config.get('intercept_distance', 25.0)
        self.defend_speed = self.config.get('defend_speed', 0.8)
        self.min_opponent_distance = self.config.get('min_opponent_distance', 5.0)
        self.target_opponent = None
        self.defend_position = None
        
    def initiate(self, state: Dict[str, Any]) -> bool:
        """
        Initiate if opponents are within defend radius.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to initiate this option
        """
        opponent_positions = state.get("opponent_positions", [])
        agent_position = state.get("agent_position", None)
        team_flag_position = state.get("team_flag_position", None)
        
        if not opponent_positions or agent_position is None or team_flag_position is None:
            return False
            
        # Check if any opponent is within defend radius
        for opp_pos in opponent_positions:
            if np.linalg.norm(np.array(opp_pos) - np.array(team_flag_position)) < self.defend_radius:
                self.target_opponent = np.array(opp_pos)
                self.defend_position = np.array(team_flag_position)
                return True
                
        return False
        
    def terminate(self, state: Dict[str, Any]) -> bool:
        """
        Terminate if no opponents are nearby or if we're too far from defend position.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to terminate this option
        """
        if self.target_opponent is None or self.defend_position is None:
            return True
            
        agent_position = state.get("agent_position", None)
        if agent_position is None:
            return True
            
        # Check if opponent is still within defend radius
        opponent_positions = state.get("opponent_positions", [])
        if not any(
            np.linalg.norm(np.array(opp_pos) - self.defend_position) < self.defend_radius
            for opp_pos in opponent_positions
        ):
            return True
            
        # Check if we're too far from defend position
        distance = np.linalg.norm(np.array(agent_position) - self.defend_position)
        if distance > self.defend_radius * 1.5:
            return True
            
        return False
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get action to defend against opponents.
        
        Args:
            state: Current environment state
            
        Returns:
            np.ndarray: Action to take [speed, heading]
        """
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is None or self.defend_position is None:
            return np.zeros(2)
            
        # Update target opponent if needed
        opponent_positions = state.get("opponent_positions", [])
        if opponent_positions:
            nearest_opp = min(
                opponent_positions,
                key=lambda pos: np.linalg.norm(np.array(pos) - self.defend_position)
            )
            self.target_opponent = np.array(nearest_opp)
            
        # Calculate intercept position
        if self.target_opponent is not None:
            direction = self.target_opponent - self.defend_position
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction = direction / distance
                intercept_position = self.defend_position + direction * self.intercept_distance
                
                # Move to intercept position
                move_direction = intercept_position - agent_position
                move_distance = np.linalg.norm(move_direction)
                
                if move_distance > 0:
                    move_direction = move_direction / move_distance
                    
                    # Calculate speed and heading
                    speed = min(1.0, move_distance / 10.0) * self.defend_speed
                    heading = np.arctan2(move_direction[1], move_direction[0]) * 180 / np.pi
                    
                    return np.array([speed, heading])
                    
        return np.zeros(2)
        
    def get_reward(self, state: Dict[str, Any], action: np.ndarray, next_state: Dict[str, Any]) -> float:
        """
        Get reward based on defense effectiveness.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            float: Reward value
        """
        reward = 0.0
        
        # Reward for maintaining defend position
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is not None and self.defend_position is not None:
            distance = np.linalg.norm(agent_position - self.defend_position)
            if distance <= self.defend_radius:
                reward += 1.0
                
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
            
        # Penalty for being too close to opponents
        for opp_pos in opponent_positions:
            opp_pos = np.array(opp_pos)
            if np.linalg.norm(opp_pos - agent_position) < self.min_opponent_distance:
                reward -= 10.0
                
        return reward
        
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """
        Update defense strategy based on opponent positions.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Update defend position if needed
        team_flag_position = state.get("team_flag_position", None)
        if team_flag_position is not None:
            self.defend_position = np.array(team_flag_position)
            
        # Adjust defend speed based on success
        if reward > 0:
            self.defend_speed = min(1.0, self.defend_speed + 0.01)
        else:
            self.defend_speed = max(0.5, self.defend_speed - 0.01)
            
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.target_opponent = None
        self.defend_position = None 