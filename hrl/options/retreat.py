import numpy as np
from typing import Dict, Any, Optional
from hrl.options.base import BaseOption

class RetreatOption(BaseOption):
    """Option for retreating to base when in danger."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the retreat option.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - danger_radius: Radius to consider opponents as dangerous (default: 20.0)
                - retreat_speed: Speed multiplier when retreating (default: 1.0)
                - safe_radius: Radius around base considered safe (default: 15.0)
                - min_health: Minimum health to trigger retreat (default: 0.3)
        """
        super().__init__("retreat", config)
        self.danger_radius = self.config.get('danger_radius', 20.0)
        self.retreat_speed = self.config.get('retreat_speed', 1.0)
        self.safe_radius = self.config.get('safe_radius', 15.0)
        self.min_health = self.config.get('min_health', 0.3)
        self.base_position = None
        self.last_position = None
        self.steps_without_progress = 0
        
    def initiate(self, state: Dict[str, Any]) -> bool:
        """
        Initiate if we're tagged, in danger, or low on health.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to initiate this option
        """
        opponent_positions = state.get("opponent_positions", [])
        agent_position = state.get("agent_position", None)
        base_position = state.get("base_position", None)
        agent_health = state.get("agent_health", 1.0)
        agent_is_tagged = state.get("agent_is_tagged", False)
        
        if agent_position is None or base_position is None:
            return False
            
        agent_pos = np.array(agent_position)
        self.base_position = np.array(base_position)
        
        # Check if we're tagged
        if agent_is_tagged:
            self.last_position = agent_pos
            self.steps_without_progress = 0
            return True
            
        # Check if we're in danger
        if opponent_positions:
            nearest_opp = min(
                opponent_positions,
                key=lambda pos: np.linalg.norm(np.array(pos) - agent_pos)
            )
            distance = np.linalg.norm(np.array(nearest_opp) - agent_pos)
            if distance < self.danger_radius:
                self.last_position = agent_pos
                self.steps_without_progress = 0
                return True
                
        # Check if we're low on health
        if agent_health < self.min_health:
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
        if self.base_position is None:
            return True
            
        agent_position = state.get("agent_position", None)
        if agent_position is None:
            return True
            
        # Check if we're stuck
        if self.steps_without_progress >= 50:
            return True
            
        # Check if we're safe
        distance_to_base = np.linalg.norm(self.base_position - np.array(agent_position))
        if distance_to_base <= self.safe_radius:
            return True
            
        # Check if we're no longer in danger
        opponent_positions = state.get("opponent_positions", [])
        agent_health = state.get("agent_health", 1.0)
        agent_is_tagged = state.get("agent_is_tagged", False)
        
        if not agent_is_tagged and agent_health >= self.min_health:
            if opponent_positions:
                nearest_opp = min(
                    opponent_positions,
                    key=lambda pos: np.linalg.norm(np.array(pos) - np.array(agent_position))
                )
                distance = np.linalg.norm(np.array(nearest_opp) - np.array(agent_position))
                if distance >= self.danger_radius:
                    return True
                    
        return False
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get action to retreat to base.
        
        Args:
            state: Current environment state
            
        Returns:
            np.ndarray: Action to take [speed, heading]
        """
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is None or self.base_position is None:
            return np.zeros(2)
            
        # Move towards base
        direction = self.base_position - agent_position
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
        speed = min(1.0, distance / 10.0) * self.retreat_speed
        heading = np.arctan2(direction[1], direction[0]) * 180 / np.pi
        
        return np.array([speed, heading])
        
    def get_reward(self, state: Dict[str, Any], action: np.ndarray, next_state: Dict[str, Any]) -> float:
        """
        Get reward based on retreat effectiveness.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            float: Reward value
        """
        reward = 0.0
        
        # Reward for getting closer to base
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is not None and self.base_position is not None:
            current_distance = np.linalg.norm(self.base_position - agent_position)
            if current_distance < self.safe_radius:
                reward += 5.0
                
        # Large reward for reaching base
        if next_state.get("agent_at_base", False):
            reward += 20.0
            
        # Penalty for being tagged
        if next_state.get("agent_is_tagged", False):
            reward -= 30.0
            
        # Penalty for getting further from base
        next_agent_pos = np.array(next_state.get("agent_position", [0, 0]))
        next_distance = np.linalg.norm(self.base_position - next_agent_pos)
        if next_distance > current_distance:
            reward -= 2.0
            
        return reward
        
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """
        Update retreat strategy based on effectiveness.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Adjust retreat speed based on success
        if reward > 0:
            self.retreat_speed = min(1.2, self.retreat_speed + 0.01)
        else:
            self.retreat_speed = max(0.8, self.retreat_speed - 0.01)
            
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.base_position = None
        self.last_position = None
        self.steps_without_progress = 0 