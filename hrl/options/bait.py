import numpy as np
from typing import Dict, Any, Optional
from hrl.options.base import BaseOption

class BaitOption(BaseOption):
    """Option for luring opponents into traps or away from objectives."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the bait option.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - bait_radius: Radius to search for baiting opportunities (default: 20.0)
                - bait_speed: Speed multiplier when baiting (default: 0.8)
                - min_opponent_distance: Minimum distance to maintain from opponents (default: 5.0)
                - max_teammate_distance: Maximum distance to maintain from teammates (default: 30.0)
                - trap_radius: Radius around trap position (default: 15.0)
        """
        super().__init__("bait", config)
        self.bait_radius = self.config.get('bait_radius', 20.0)
        self.bait_speed = self.config.get('bait_speed', 0.8)
        self.min_opponent_distance = self.config.get('min_opponent_distance', 5.0)
        self.max_teammate_distance = self.config.get('max_teammate_distance', 30.0)
        self.trap_radius = self.config.get('trap_radius', 15.0)
        self.target_opponent = None
        self.trap_position = None
        self.bait_position = None
        self.last_position = None
        self.steps_without_progress = 0
        
    def initiate(self, state: Dict[str, Any]) -> bool:
        """
        Initiate if we see an opportunity to bait an opponent.
        
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
        
        # Find opponent that can be baited
        for opp_pos in opponent_positions:
            opp_pos = np.array(opp_pos)
            
            # Check if opponent is near our flag or an objective
            if np.linalg.norm(opp_pos - np.array(state.get("team_flag_position", [0, 0]))) < self.bait_radius:
                # Find a good trap position near teammates
                for teammate_pos in teammate_positions:
                    teammate_pos = np.array(teammate_pos)
                    if np.linalg.norm(teammate_pos - opp_pos) < self.max_teammate_distance:
                        self.target_opponent = opp_pos
                        self.trap_position = teammate_pos
                        self.bait_position = self._calculate_bait_position(agent_pos, opp_pos, teammate_pos)
                        self.last_position = agent_pos
                        self.steps_without_progress = 0
                        return True
                        
        return False
        
    def terminate(self, state: Dict[str, Any]) -> bool:
        """
        Terminate if baiting opportunity is lost or if we're too far from target.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to terminate this option
        """
        if self.target_opponent is None or self.bait_position is None:
            return True
            
        agent_position = state.get("agent_position", None)
        if agent_position is None:
            return True
            
        # Check if we're stuck
        if self.steps_without_progress >= 50:
            return True
            
        # Check if opponent is still following
        opponent_positions = state.get("opponent_positions", [])
        if not any(
            np.linalg.norm(np.array(opp_pos) - np.array(agent_position)) < self.bait_radius
            for opp_pos in opponent_positions
        ):
            return True
            
        return False
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get action to execute baiting maneuver.
        
        Args:
            state: Current environment state
            
        Returns:
            np.ndarray: Action to take [speed, heading]
        """
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is None or self.bait_position is None:
            return np.zeros(2)
            
        # Update bait position based on opponent movement
        opponent_positions = state.get("opponent_positions", [])
        if opponent_positions:
            # Find the closest opponent
            closest_opp = min(
                opponent_positions,
                key=lambda pos: np.linalg.norm(np.array(pos) - agent_position)
            )
            self.target_opponent = np.array(closest_opp)
            
            # Adjust bait position to lead opponent towards trap
            direction_to_trap = self.trap_position - self.target_opponent
            if np.linalg.norm(direction_to_trap) > 0:
                direction_to_trap = direction_to_trap / np.linalg.norm(direction_to_trap)
                self.bait_position = self.target_opponent + direction_to_trap * self.min_opponent_distance
                
        # Move to bait position
        direction = self.bait_position - agent_position
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
        speed = min(1.0, distance / 10.0) * self.bait_speed
        heading = np.arctan2(direction[1], direction[0]) * 180 / np.pi
        
        return np.array([speed, heading])
        
    def get_reward(self, state: Dict[str, Any], action: np.ndarray, next_state: Dict[str, Any]) -> float:
        """
        Get reward based on baiting effectiveness.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            float: Reward value
        """
        reward = 0.0
        
        # Reward for maintaining bait position
        agent_position = np.array(state.get("agent_position", [0, 0]))
        if agent_position is not None and self.bait_position is not None:
            distance = np.linalg.norm(agent_position - self.bait_position)
            if distance < 2.0:
                reward += 1.0
                
        # Reward for opponent following
        opponent_positions = state.get("opponent_positions", [])
        if opponent_positions:
            for opp_pos in opponent_positions:
                opp_pos = np.array(opp_pos)
                if np.linalg.norm(opp_pos - agent_position) < self.bait_radius:
                    reward += 2.0
                    
        # Large reward for opponent entering trap
        if self.trap_position is not None:
            for opp_pos in opponent_positions:
                opp_pos = np.array(opp_pos)
                if np.linalg.norm(opp_pos - self.trap_position) < self.trap_radius:
                    reward += 50.0
                    
        # Large reward for teammate tagging opponent in trap
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
        Update baiting strategy based on opponent behavior.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Adjust bait speed based on success
        if reward > 0:
            self.bait_speed = min(1.0, self.bait_speed + 0.01)
        else:
            self.bait_speed = max(0.6, self.bait_speed - 0.01)
            
    def _calculate_bait_position(self, agent_pos: np.ndarray, opp_pos: np.ndarray, trap_pos: np.ndarray) -> np.ndarray:
        """Calculate optimal baiting position."""
        # Calculate direction from opponent to trap
        direction = trap_pos - opp_pos
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            
        # Position ourselves between opponent and trap
        bait_distance = self.min_opponent_distance * 1.2
        return opp_pos + direction * bait_distance
        
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.target_opponent = None
        self.trap_position = None
        self.bait_position = None
        self.last_position = None
        self.steps_without_progress = 0 