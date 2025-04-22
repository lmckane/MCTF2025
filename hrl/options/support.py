from typing import Dict, Any, Optional
import numpy as np
from hrl.options.base import BaseOption

class SupportOption(BaseOption):
    """Option for supporting teammates by providing backup and assistance."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the support option.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - support_distance: Distance to maintain from supported teammate (default: 10.0)
                - backup_distance: Distance at which to provide backup (default: 20.0)
                - assist_speed: Speed multiplier when assisting (default: 0.9)
                - min_teammate_health: Minimum teammate health to trigger support (default: 0.5)
        """
        super().__init__("support", config)
        self.support_distance = self.config.get('support_distance', 10.0)
        self.backup_distance = self.config.get('backup_distance', 20.0)
        self.assist_speed = self.config.get('assist_speed', 0.9)
        self.min_teammate_health = self.config.get('min_teammate_health', 0.5)
        self.target_teammate = None
        self.support_position = None
        
    def initiate(self, state: Dict[str, Any]) -> bool:
        """
        Initiate if a teammate needs support (low health or engaged with opponent).
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to initiate this option
        """
        teammate_states = state.get("teammate_states", [])
        opponent_positions = state.get("opponent_positions", [])
        agent_position = state.get("agent_position", None)
        
        if not teammate_states or agent_position is None:
            return False
            
        # Find teammate that needs support
        for teammate in teammate_states:
            teammate_pos = teammate.get("position", None)
            teammate_health = teammate.get("health", 1.0)
            
            if teammate_pos is None:
                continue
                
            # Check if teammate is in danger or needs backup
            needs_support = False
            
            # Check health
            if teammate_health < self.min_teammate_health:
                needs_support = True
                
            # Check if engaged with opponent
            for opp_pos in opponent_positions:
                if np.linalg.norm(np.array(opp_pos) - np.array(teammate_pos)) < self.backup_distance:
                    needs_support = True
                    break
                    
            if needs_support:
                self.target_teammate = teammate
                self.support_position = np.array(teammate_pos)
                return True
                
        return False
        
    def terminate(self, state: Dict[str, Any]) -> bool:
        """
        Terminate if no teammates need support or if we're too far from target.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to terminate this option
        """
        if self.target_teammate is None or self.support_position is None:
            return True
            
        agent_position = state.get("agent_position", None)
        if agent_position is None:
            return True
            
        # Check if teammate still needs support
        teammate_health = self.target_teammate.get("health", 1.0)
        if teammate_health >= self.min_teammate_health:
            return True
            
        # Check if we're too far from target
        distance = np.linalg.norm(np.array(agent_position) - self.support_position)
        if distance > self.backup_distance * 1.5:
            return True
            
        return False
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get action to support the target teammate.
        
        Args:
            state: Current environment state
            
        Returns:
            np.ndarray: Action to take [speed, heading]
        """
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is None or self.support_position is None:
            return np.zeros(2)
            
        # Calculate optimal support position
        opponent_positions = state.get("opponent_positions", [])
        if opponent_positions:
            # Position ourselves between teammate and nearest opponent
            nearest_opp = min(
                opponent_positions,
                key=lambda pos: np.linalg.norm(np.array(pos) - self.support_position)
            )
            direction = np.array(nearest_opp) - self.support_position
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction = direction / distance
                self.support_position = self.support_position + direction * self.support_distance
                
        # Move to support position
        direction = self.support_position - agent_position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance
            
        # Calculate speed and heading
        speed = min(1.0, distance / 10.0) * self.assist_speed
        heading = np.arctan2(direction[1], direction[0]) * 180 / np.pi
        
        return np.array([speed, heading])
        
    def get_reward(self, state: Dict[str, Any], action: np.ndarray, next_state: Dict[str, Any]) -> float:
        """
        Get reward based on support effectiveness.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            float: Reward value
        """
        reward = 0.0
        
        # Reward for maintaining support distance
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is not None and self.support_position is not None:
            distance = np.linalg.norm(agent_position - self.support_position)
            if abs(distance - self.support_distance) < 2.0:
                reward += 1.0
                
        # Reward for teammate health improvement
        if self.target_teammate is not None:
            current_health = self.target_teammate.get("health", 1.0)
            next_health = next_state.get("teammate_health", {}).get(
                self.target_teammate.get("id", ""), current_health
            )
            if next_health > current_health:
                reward += 5.0
                
        # Reward for protecting teammate from opponents
        opponent_positions = state.get("opponent_positions", [])
        for opp_pos in opponent_positions:
            opp_pos = np.array(opp_pos)
            if np.linalg.norm(opp_pos - agent_position) < self.support_distance:
                reward += 2.0
                
        # Penalty for being too far from teammate
        if distance > self.backup_distance:
            reward -= 2.0
            
        return reward
        
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """
        Update support strategy based on teammate state.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Update support position based on teammate movement
        if self.target_teammate is not None:
            teammate_pos = self.target_teammate.get("position", None)
            if teammate_pos is not None:
                self.support_position = np.array(teammate_pos)
                
        # Adjust assist speed based on success
        if reward > 0:
            self.assist_speed = min(1.0, self.assist_speed + 0.01)
        else:
            self.assist_speed = max(0.5, self.assist_speed - 0.01)
            
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.target_teammate = None
        self.support_position = None 