from typing import Dict, Any, List
import numpy as np

class RewardShaper:
    """Shapes and modifies rewards to improve learning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.discount_factor = config.get("discount_factor", 0.99)
        self.reward_scale = config.get("reward_scale", 1.0)
        self.potential_based = config.get("potential_based", True)
        
    def shape_reward(self, state: Dict[str, Any], action: np.ndarray,
                    reward: float, next_state: Dict[str, Any]) -> float:
        """
        Shape the reward based on state and action.
        
        Args:
            state: Current state
            action: Action taken
            reward: Original reward
            next_state: Next state
            
        Returns:
            float: Shaped reward
        """
        shaped_reward = reward * self.reward_scale
        
        if self.potential_based:
            # Add potential-based shaping
            potential = self._calculate_potential(next_state)
            prev_potential = self._calculate_potential(state)
            shaped_reward += self.discount_factor * potential - prev_potential
            
        # Add additional shaping terms
        shaped_reward += self._get_distance_reward(state, next_state)
        shaped_reward += self._get_safety_reward(state, next_state)
        shaped_reward += self._get_progress_reward(state, next_state)
        
        return shaped_reward
        
    def _calculate_potential(self, state: Dict[str, Any]) -> float:
        """Calculate potential for potential-based shaping."""
        potential = 0.0
        
        # Distance to goal
        if "goal_position" in state:
            dist = self._get_distance(state["agent_position"], state["goal_position"])
            potential -= dist
            
        # Flag capture potential
        if "has_flag" in state and state["has_flag"]:
            potential += 10.0
            
        # Defense potential
        if "flag_position" in state:
            dist = self._get_distance(state["agent_position"], state["flag_position"])
            potential += 5.0 / (1.0 + dist)
            
        return potential
        
    def _get_distance_reward(self, state: Dict[str, Any], next_state: Dict[str, Any]) -> float:
        """Get reward based on distance changes."""
        reward = 0.0
        
        # Reward for moving towards goal
        if "goal_position" in state:
            prev_dist = self._get_distance(state["agent_position"], state["goal_position"])
            next_dist = self._get_distance(next_state["agent_position"], state["goal_position"])
            reward += (prev_dist - next_dist) * 0.1
            
        # Reward for moving away from opponents
        if "opponent_positions" in state:
            prev_min_dist = min(self._get_distance(state["agent_position"], opp_pos)
                              for opp_pos in state["opponent_positions"])
            next_min_dist = min(self._get_distance(next_state["agent_position"], opp_pos)
                              for opp_pos in state["opponent_positions"])
            reward += (next_min_dist - prev_min_dist) * 0.05
            
        return reward
        
    def _get_safety_reward(self, state: Dict[str, Any], next_state: Dict[str, Any]) -> float:
        """Get reward based on safety considerations."""
        reward = 0.0
        
        # Penalty for being tagged
        if "is_tagged" in next_state and next_state["is_tagged"]:
            reward -= 1.0
            
        # Penalty for being close to opponents
        if "opponent_positions" in state:
            min_dist = min(self._get_distance(next_state["agent_position"], opp_pos)
                          for opp_pos in state["opponent_positions"])
            if min_dist < 1.0:
                reward -= 0.5
                
        return reward
        
    def _get_progress_reward(self, state: Dict[str, Any], next_state: Dict[str, Any]) -> float:
        """Get reward based on task progress."""
        reward = 0.0
        
        # Reward for capturing flag
        if "has_flag" in next_state and next_state["has_flag"] and not state.get("has_flag", False):
            reward += 5.0
            
        # Reward for returning flag
        if "flag_returned" in next_state and next_state["flag_returned"]:
            reward += 10.0
            
        # Reward for tagging opponents
        if "opponents_tagged" in next_state:
            reward += next_state["opponents_tagged"] * 2.0
            
        return reward
        
    def _get_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate distance between two positions."""
        return np.linalg.norm(pos1 - pos2) 