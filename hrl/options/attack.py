from typing import Dict, Any, Optional
import numpy as np
from hrl.options.base import BaseOption

class AttackOption(BaseOption):
    """Option for aggressive play against opponents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the attack option.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - attack_radius: Radius to search for opponents (default: 30.0)
                - min_attack_distance: Minimum distance to maintain from opponents (default: 3.0)
                - attack_speed: Speed multiplier when attacking (default: 1.0)
                - max_stuck_steps: Maximum steps without progress before termination (default: 50)
        """
        super().__init__("attack", config)
        self.attack_radius = self.config.get('attack_radius', 30.0)
        self.min_attack_distance = self.config.get('min_attack_distance', 3.0)
        self.attack_speed = self.config.get('attack_speed', 1.0)
        self.max_stuck_steps = self.config.get('max_stuck_steps', 50)
        self.target_opponent = None
        self.last_position = None
        self.steps_without_progress = 0
        
    def initiate(self, state: Dict[str, Any]) -> bool:
        """
        Initiate if we see an opponent within attack radius.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to initiate this option
        """
        opponent_positions = state.get("opponent_positions", [])
        agent_position = state.get("agent_position", None)
        agent_has_flag = state.get("agent_has_flag", False)
        
        if not opponent_positions or agent_position is None or agent_has_flag:
            return False
            
        # Find nearest opponent within attack radius
        agent_pos = np.array(agent_position)
        for opp_pos in opponent_positions:
            opp_pos = np.array(opp_pos)
            distance = np.linalg.norm(opp_pos - agent_pos)
            if distance < self.attack_radius:
                self.target_opponent = opp_pos
                self.last_position = agent_pos
                self.steps_without_progress = 0
                return True
                
        return False
        
    def terminate(self, state: Dict[str, Any]) -> bool:
        """
        Terminate if we lose sight of opponent, get stuck, or get the flag.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to terminate this option
        """
        agent_has_flag = state.get("agent_has_flag", False)
        agent_position = state.get("agent_position", None)
        
        if agent_has_flag or agent_position is None:
            return True
            
        # Check if we're stuck
        if self.steps_without_progress >= self.max_stuck_steps:
            return True
            
        # Check if opponent is still within attack radius
        opponent_positions = state.get("opponent_positions", [])
        if not any(
            np.linalg.norm(np.array(opp_pos) - np.array(agent_position)) < self.attack_radius
            for opp_pos in opponent_positions
        ):
            return True
            
        return False
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get action to attack the target opponent.
        
        Args:
            state: Current environment state
            
        Returns:
            np.ndarray: Action to take [speed, heading]
        """
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is None or self.target_opponent is None:
            return np.zeros(2)
            
        # Update target opponent if needed
        opponent_positions = state.get("opponent_positions", [])
        if opponent_positions:
            nearest_opp = min(
                opponent_positions,
                key=lambda pos: np.linalg.norm(np.array(pos) - agent_position)
            )
            self.target_opponent = np.array(nearest_opp)
            
        # Calculate direction to target
        direction = self.target_opponent - agent_position
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
        speed = min(1.0, distance / 10.0) * self.attack_speed
        heading = np.arctan2(direction[1], direction[0]) * 180 / np.pi
        
        return np.array([speed, heading])
        
    def get_reward(self, state: Dict[str, Any], action: np.ndarray, next_state: Dict[str, Any]) -> float:
        """
        Get reward based on attack effectiveness.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            float: Reward value
        """
        reward = 0.0
        
        # Reward for getting closer to opponent
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is not None and self.target_opponent is not None:
            current_dist = np.linalg.norm(agent_position - self.target_opponent)
            next_dist = np.linalg.norm(
                np.array(next_state.get("agent_position", agent_position)) - self.target_opponent
            )
            reward += (current_dist - next_dist) * 5.0
            
        # Large reward for tagging opponent
        if next_state.get("opponent_tagged", False):
            reward += 50.0
            
        # Penalty for being tagged
        if next_state.get("agent_is_tagged", False):
            reward -= 30.0
            
        # Penalty for being too close to opponent
        if current_dist < self.min_attack_distance:
            reward -= 5.0
            
        return reward
        
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """
        Update attack strategy based on opponent behavior.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Update target opponent if needed
        opponent_positions = state.get("opponent_positions", [])
        if opponent_positions:
            agent_position = np.array(state.get("agent_position", [0, 0]))
            nearest_opp = min(
                opponent_positions,
                key=lambda pos: np.linalg.norm(np.array(pos) - agent_position)
            )
            self.target_opponent = np.array(nearest_opp)
            
        # Adjust attack speed based on success
        if reward > 0:
            self.attack_speed = min(1.2, self.attack_speed + 0.01)
        else:
            self.attack_speed = max(0.8, self.attack_speed - 0.01)
            
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.target_opponent = None
        self.last_position = None
        self.steps_without_progress = 0 