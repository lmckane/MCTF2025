from typing import Dict, Any, Optional
import numpy as np
from hrl.options.base import BaseOption

class CaptureFlagOption(BaseOption):
    """Option for capturing the opponent's flag."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the capture flag option.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - approach_speed: Speed multiplier when approaching flag (default: 1.0)
                - min_distance: Minimum distance to maintain from flag (default: 2.0)
                - path_planning: Whether to use path planning (default: False)
                - path_update_freq: How often to update path (default: 10)
        """
        super().__init__("capture_flag", config)
        self.target_flag = None
        self.path_to_flag = []
        self.approach_speed = self.config.get('approach_speed', 1.0)
        self.min_distance = self.config.get('min_distance', 2.0)
        self.use_path_planning = self.config.get('path_planning', False)
        self.path_update_freq = self.config.get('path_update_freq', 10)
        self.steps_since_path_update = 0
        
    def initiate(self, state: Dict[str, Any]) -> bool:
        """
        Initiate if we don't have our flag and the opponent's flag is reachable.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to initiate this option
        """
        agent_has_flag = state.get("agent_has_flag", False)
        agent_is_tagged = state.get("agent_is_tagged", False)
        opponent_flag_position = state.get("opponent_flag_position", None)
        agent_position = state.get("agent_position", None)
        
        if not agent_has_flag and not agent_is_tagged and opponent_flag_position is not None and agent_position is not None:
            self.target_flag = np.array(opponent_flag_position)
            if self.use_path_planning:
                self._update_path_planning(agent_position)
            return True
        return False
        
    def terminate(self, state: Dict[str, Any]) -> bool:
        """
        Terminate if we have the flag or if we're tagged.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to terminate this option
        """
        agent_has_flag = state.get("agent_has_flag", False)
        agent_is_tagged = state.get("agent_is_tagged", False)
        return agent_has_flag or agent_is_tagged
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get action to move towards the opponent's flag.
        
        Args:
            state: Current environment state
            
        Returns:
            np.ndarray: Action to take [speed, heading]
        """
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is None or self.target_flag is None:
            return np.zeros(2)
            
        # Update path planning if needed
        if self.use_path_planning and self.steps_since_path_update >= self.path_update_freq:
            self._update_path_planning(agent_position)
            self.steps_since_path_update = 0
            
        # Calculate direction to target
        if self.use_path_planning and self.path_to_flag:
            # Use next point in path
            target = self.path_to_flag[0]
            if np.linalg.norm(agent_position - target) < 0.5:
                self.path_to_flag.pop(0)
                if self.path_to_flag:
                    target = self.path_to_flag[0]
        else:
            # Direct path to flag
            target = self.target_flag
            
        direction = target - agent_position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance
            
        # Calculate speed and heading
        speed = min(1.0, distance / 10.0) * self.approach_speed
        heading = np.arctan2(direction[1], direction[0]) * 180 / np.pi
        
        return np.array([speed, heading])
        
    def get_reward(self, state: Dict[str, Any], action: np.ndarray, next_state: Dict[str, Any]) -> float:
        """
        Get reward based on progress towards flag and successful capture.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            float: Reward value
        """
        reward = 0.0
        
        # Reward for getting closer to flag
        current_pos = np.array(state.get("agent_position", [0, 0]))
        next_pos = np.array(next_state.get("agent_position", [0, 0]))
        current_dist = np.linalg.norm(current_pos - self.target_flag)
        next_dist = np.linalg.norm(next_pos - self.target_flag)
        reward += (current_dist - next_dist) * 10.0
        
        # Large reward for capturing flag
        if next_state.get("agent_has_flag", False):
            reward += 100.0
            
        # Penalty for being tagged
        if next_state.get("agent_is_tagged", False):
            reward -= 50.0
            
        # Small penalty for being too close to opponents
        opponent_positions = next_state.get("opponent_positions", [])
        for opp_pos in opponent_positions:
            dist = np.linalg.norm(next_pos - np.array(opp_pos))
            if dist < 5.0:  # Hard-coded danger distance
                reward -= 5.0
                
        return reward
        
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """
        Update internal state and adjust parameters based on experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.steps_since_path_update += 1
        
        # Update path planning if needed
        if self.use_path_planning and self.steps_since_path_update >= self.path_update_freq:
            agent_position = np.array(next_state.get("agent_position", [0, 0]))
            self._update_path_planning(agent_position)
            self.steps_since_path_update = 0
            
        # Adjust approach speed based on success
        if reward > 0:
            self.approach_speed = min(1.5, self.approach_speed + 0.01)
        else:
            self.approach_speed = max(0.5, self.approach_speed - 0.01)
            
    def _update_path_planning(self, agent_position: np.ndarray):
        """
        Update the planned path to the flag.
        
        Args:
            agent_position: Current agent position
        """
        if self.target_flag is None:
            return
            
        # Simple path planning: move directly towards flag
        direction = self.target_flag - agent_position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance
            # Create path points
            num_points = min(10, int(distance / 5))
            self.path_to_flag = [
                agent_position + direction * (i * distance / num_points)
                for i in range(1, num_points + 1)
            ]
            
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.target_flag = None
        self.path_to_flag = []
        self.steps_since_path_update = 0 