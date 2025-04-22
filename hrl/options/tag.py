import numpy as np
from typing import Dict, Any, Optional
from hrl.options.base import BaseOption

class TagOption(BaseOption):
    """Option for tagging intruders in territory."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the tag option.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - tag_radius: Radius to search for intruders (default: 30.0)
                - tag_speed: Speed multiplier when tagging (default: 1.0)
                - min_tag_distance: Minimum distance to maintain from intruder (default: 5.0)
                - territory_radius: Radius of territory to defend (default: 40.0)
        """
        super().__init__("tag", config)
        self.tag_radius = self.config.get('tag_radius', 30.0)
        self.tag_speed = self.config.get('tag_speed', 1.0)
        self.min_tag_distance = self.config.get('min_tag_distance', 5.0)
        self.territory_radius = self.config.get('territory_radius', 40.0)
        self.target_intruder = None
        self.last_position = None
        self.steps_without_progress = 0
        
    def initiate(self, state: Dict[str, Any]) -> bool:
        """
        Initiate if there's an intruder in our territory.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to initiate this option
        """
        opponent_positions = state.get("opponent_positions", [])
        agent_position = state.get("agent_position", None)
        base_position = state.get("base_position", None)
        
        if not opponent_positions or agent_position is None or base_position is None:
            return False
            
        agent_pos = np.array(agent_position)
        base_pos = np.array(base_position)
        
        # Find intruder in territory
        for opp_pos in opponent_positions:
            opp_pos = np.array(opp_pos)
            distance_to_base = np.linalg.norm(opp_pos - base_pos)
            
            if distance_to_base < self.territory_radius:
                self.target_intruder = opp_pos
                self.last_position = agent_pos
                self.steps_without_progress = 0
                return True
                
        return False
        
    def terminate(self, state: Dict[str, Any]) -> bool:
        """
        Terminate if intruder is gone or we're stuck.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to terminate this option
        """
        if self.target_intruder is None:
            return True
            
        agent_position = state.get("agent_position", None)
        if agent_position is None:
            return True
            
        # Check if we're stuck
        if self.steps_without_progress >= 50:
            return True
            
        # Check if intruder is still in territory
        base_position = state.get("base_position", None)
        if base_position is not None:
            distance_to_base = np.linalg.norm(self.target_intruder - np.array(base_position))
            if distance_to_base >= self.territory_radius:
                return True
                
        return False
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get action to tag the intruder.
        
        Args:
            state: Current environment state
            
        Returns:
            np.ndarray: Action to take [speed, heading]
        """
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is None or self.target_intruder is None:
            return np.zeros(2)
            
        # Update target position based on intruder movement
        opponent_positions = state.get("opponent_positions", [])
        if opponent_positions:
            # Find the intruder closest to our base
            base_position = state.get("base_position", None)
            if base_position is not None:
                base_pos = np.array(base_position)
                intruder = min(
                    opponent_positions,
                    key=lambda pos: np.linalg.norm(np.array(pos) - base_pos)
                )
                self.target_intruder = np.array(intruder)
                
        # Move towards intruder
        direction = self.target_intruder - agent_position
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
        speed = min(1.0, distance / 10.0) * self.tag_speed
        heading = np.arctan2(direction[1], direction[0]) * 180 / np.pi
        
        return np.array([speed, heading])
        
    def get_reward(self, state: Dict[str, Any], action: np.ndarray, next_state: Dict[str, Any]) -> float:
        """
        Get reward based on tagging effectiveness.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            float: Reward value
        """
        reward = 0.0
        
        # Reward for getting closer to intruder
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is not None and self.target_intruder is not None:
            current_distance = np.linalg.norm(self.target_intruder - agent_position)
            if current_distance < self.min_tag_distance:
                reward += 2.0
                
        # Large reward for tagging intruder
        if next_state.get("opponent_tagged", False):
            reward += 50.0
            
        # Penalty for being tagged
        if next_state.get("agent_is_tagged", False):
            reward -= 30.0
            
        # Penalty for intruder getting away
        base_position = state.get("base_position", None)
        if base_position is not None:
            current_distance = np.linalg.norm(self.target_intruder - np.array(base_position))
            next_distance = np.linalg.norm(self.target_intruder - np.array(next_state.get("base_position", [0, 0])))
            if next_distance > current_distance:
                reward -= 5.0
                
        return reward
        
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """
        Update tagging strategy based on effectiveness.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Adjust tag speed based on success
        if reward > 0:
            self.tag_speed = min(1.2, self.tag_speed + 0.01)
        else:
            self.tag_speed = max(0.8, self.tag_speed - 0.01)
            
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.target_intruder = None
        self.last_position = None
        self.steps_without_progress = 0 