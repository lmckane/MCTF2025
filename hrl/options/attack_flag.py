import numpy as np
from typing import Dict, Any
from hrl.options.base import BaseOption

class AttackFlagOption(BaseOption):
    """Option for attacking and capturing the opponent's flag."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the attack flag option.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("attack_flag")
        self.config = config
        self.target_position = None
        self.last_position = None
        self.steps_without_progress = 0
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get the action to move towards and capture the opponent's flag.
        
        Args:
            state: Current environment state
            
        Returns:
            np.ndarray: Action to take
        """
        # Get current position
        current_pos = np.array(state['agent_position'])
        
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
        speed = min(1.0, distance / 10.0)  # Slow down as we get closer
        heading = np.arctan2(direction[1], direction[0]) * 180 / np.pi
        
        return np.array([speed, heading])
    
    def is_termination_condition_met(self, state: Dict[str, Any]) -> bool:
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
        if self.steps_without_progress > self.config.get('max_stuck_steps', 50):
            return True
            
        # Terminate if opponent is too close
        if 'opponent_position' in state:
            opponent_pos = np.array(state['opponent_position'])
            current_pos = np.array(state['agent_position'])
            distance = np.linalg.norm(opponent_pos - current_pos)
            if distance < self.config.get('evade_distance', 10.0):
                return True
                
        return False
    
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