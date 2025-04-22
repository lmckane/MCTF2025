import numpy as np
from typing import Dict, Any, Optional
from hrl.options.base import BaseOption

class AttackFlagOption(BaseOption):
    """Option for attacking and capturing the opponent's flag."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the attack flag option.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - max_stuck_steps: Maximum steps without progress before termination (default: 50)
                - evade_distance: Distance at which to evade opponents (default: 10.0)
                - approach_speed: Speed multiplier when approaching flag (default: 1.0)
                - min_distance: Minimum distance to maintain from flag (default: 2.0)
        """
        super().__init__("attack_flag", config)
        self.target_position = None
        self.last_position = None
        self.steps_without_progress = 0
        self.max_stuck_steps = self.config.get('max_stuck_steps', 50)
        self.evade_distance = self.config.get('evade_distance', 10.0)
        self.approach_speed = self.config.get('approach_speed', 1.0)
        self.min_distance = self.config.get('min_distance', 2.0)
        
    def initiate(self, state: Dict[str, Any]) -> bool:
        """
        Initiate if we don't have our flag and the opponent's flag is reachable.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to initiate this option
        """
        agent_has_flag = state.get('agent_has_flag', False)
        agent_is_tagged = state.get('agent_is_tagged', False)
        opponent_flag_position = state.get('opponent_flag_position', None)
        agent_position = state.get('agent_position', None)
        
        if not agent_has_flag and not agent_is_tagged and opponent_flag_position is not None and agent_position is not None:
            # Check if any opponents are too close
            opponent_positions = state.get('opponent_positions', [])
            for opp_pos in opponent_positions:
                if np.linalg.norm(np.array(opp_pos) - np.array(agent_position)) < self.evade_distance:
                    return False
            return True
        return False
        
    def terminate(self, state: Dict[str, Any]) -> bool:
        """
        Check if the attack flag option should terminate.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether the option should terminate
        """
        # Terminate if we have the flag
        if state.get('agent_has_flag', False):
            return True
            
        # Terminate if we're tagged
        if state.get('agent_is_tagged', False):
            return True
            
        # Terminate if we're stuck
        if self.steps_without_progress > self.max_stuck_steps:
            return True
            
        # Terminate if opponent is too close
        agent_position = state.get('agent_position', None)
        if agent_position is not None:
            opponent_positions = state.get('opponent_positions', [])
            for opp_pos in opponent_positions:
                if np.linalg.norm(np.array(opp_pos) - np.array(agent_position)) < self.evade_distance:
                    return True
                    
        return False
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get the action to move towards and capture the opponent's flag.
        
        Args:
            state: Current environment state
            
        Returns:
            np.ndarray: Action to take [speed, heading]
        """
        # Get current position
        current_pos = np.array(state.get('agent_position', [0, 0]))
        
        # If we don't have a target or need to update it
        if self.target_position is None or self._should_update_target(state):
            self.target_position = self._get_target_position(state)
            self.last_position = current_pos
            self.steps_without_progress = 0
            
        # Calculate direction to target
        direction = self.target_position - current_pos
        distance = np.linalg.norm(direction)
        
        # Normalize direction
        if distance > 0:
            direction = direction / distance
            
        # Check if we're making progress
        if self.last_position is not None:
            progress = np.linalg.norm(current_pos - self.last_position)
            if progress < 0.1:  # Threshold for considering movement
                self.steps_without_progress += 1
            else:
                self.steps_without_progress = 0
                
        self.last_position = current_pos
        
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
        current_pos = np.array(state.get('agent_position', [0, 0]))
        next_pos = np.array(next_state.get('agent_position', [0, 0]))
        current_dist = np.linalg.norm(current_pos - self.target_position)
        next_dist = np.linalg.norm(next_pos - self.target_position)
        reward += (current_dist - next_dist) * 10.0
        
        # Large reward for capturing flag
        if next_state.get('agent_has_flag', False):
            reward += 100.0
            
        # Penalty for being tagged
        if next_state.get('agent_is_tagged', False):
            reward -= 50.0
            
        # Small penalty for being too close to opponents
        opponent_positions = next_state.get('opponent_positions', [])
        for opp_pos in opponent_positions:
            dist = np.linalg.norm(next_pos - np.array(opp_pos))
            if dist < self.evade_distance:
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
        # Update target position if needed
        if self._should_update_target(next_state):
            self.target_position = self._get_target_position(next_state)
            
        # Adjust approach speed based on success
        if reward > 0:
            self.approach_speed = min(1.5, self.approach_speed + 0.01)
        else:
            self.approach_speed = max(0.5, self.approach_speed - 0.01)
            
    def _get_target_position(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get the target position to move towards.
        
        Args:
            state: Current environment state
            
        Returns:
            np.ndarray: Target position
        """
        # If opponent's flag position is known, target it
        if 'opponent_flag_position' in state:
            return np.array(state['opponent_flag_position'])
            
        # Otherwise, target the opponent's side
        field_size = state.get('env_size', [160.0, 80.0])
        return np.array([field_size[0] * 0.75, field_size[1] / 2])
        
    def _should_update_target(self, state: Dict[str, Any]) -> bool:
        """
        Check if we should update our target position.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to update target
        """
        # Update if opponent's flag position has changed
        if 'opponent_flag_position' in state:
            new_pos = np.array(state['opponent_flag_position'])
            if self.target_position is None or not np.allclose(new_pos, self.target_position):
                return True
                
        return False
        
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.target_position = None
        self.last_position = None
        self.steps_without_progress = 0 