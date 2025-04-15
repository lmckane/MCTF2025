from typing import Dict, Any, List
import numpy as np

class OptionReward:
    """Computes rewards for different options in the HRL system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reward_weights = config.get("reward_weights", {
            "flag_capture": 10.0,
            "flag_return": 20.0,
            "tag_opponent": 5.0,
            "distance_to_flag": -0.1,
            "distance_to_base": -0.1,
            "safety": -0.5
        })
        
    def compute_reward(self, option_name: str, state: Dict[str, Any],
                      next_state: Dict[str, Any]) -> float:
        """
        Compute reward for a specific option.
        
        Args:
            option_name: Name of the option
            state: Current state
            next_state: Next state
            
        Returns:
            float: Computed reward
        """
        reward = 0.0
        
        # Base rewards for all options
        reward += self._get_flag_capture_reward(state, next_state)
        reward += self._get_flag_return_reward(state, next_state)
        reward += self._get_tag_reward(state, next_state)
        
        # Option-specific rewards
        if option_name == "attack":
            reward += self._get_attack_reward(state, next_state)
        elif option_name == "defend":
            reward += self._get_defend_reward(state, next_state)
        elif option_name == "patrol":
            reward += self._get_patrol_reward(state, next_state)
        elif option_name == "evade":
            reward += self._get_evade_reward(state, next_state)
            
        return reward
        
    def _get_flag_capture_reward(self, state: Dict[str, Any],
                               next_state: Dict[str, Any]) -> float:
        """Get reward for capturing the flag."""
        if (not state.get("has_flag", False) and 
            next_state.get("has_flag", False)):
            return self.reward_weights["flag_capture"]
        return 0.0
        
    def _get_flag_return_reward(self, state: Dict[str, Any],
                              next_state: Dict[str, Any]) -> float:
        """Get reward for returning the flag."""
        if (state.get("has_flag", False) and 
            next_state.get("flag_returned", False)):
            return self.reward_weights["flag_return"]
        return 0.0
        
    def _get_tag_reward(self, state: Dict[str, Any],
                       next_state: Dict[str, Any]) -> float:
        """Get reward for tagging opponents."""
        if "opponents_tagged" in next_state:
            return (next_state["opponents_tagged"] * 
                   self.reward_weights["tag_opponent"])
        return 0.0
        
    def _get_attack_reward(self, state: Dict[str, Any],
                         next_state: Dict[str, Any]) -> float:
        """Get reward specific to attack option."""
        reward = 0.0
        
        # Reward for moving towards opponent flag
        if "flag_position" in state:
            prev_dist = self._get_distance(state["agent_position"], state["flag_position"])
            next_dist = self._get_distance(next_state["agent_position"], state["flag_position"])
            reward += (prev_dist - next_dist) * self.reward_weights["distance_to_flag"]
            
        # Penalty for being too close to opponents
        if "opponent_positions" in state:
            min_dist = min(self._get_distance(next_state["agent_position"], opp_pos)
                          for opp_pos in state["opponent_positions"])
            if min_dist < 2.0:
                reward += self.reward_weights["safety"]
                
        return reward
        
    def _get_defend_reward(self, state: Dict[str, Any],
                         next_state: Dict[str, Any]) -> float:
        """Get reward specific to defend option."""
        reward = 0.0
        
        # Reward for staying near own flag
        if "own_flag_position" in state:
            dist = self._get_distance(next_state["agent_position"], state["own_flag_position"])
            reward -= dist * self.reward_weights["distance_to_base"]
            
        # Reward for intercepting opponents
        if "opponent_positions" in state:
            min_dist = min(self._get_distance(next_state["agent_position"], opp_pos)
                          for opp_pos in state["opponent_positions"])
            if min_dist < 5.0:
                reward += 1.0
                
        return reward
        
    def _get_patrol_reward(self, state: Dict[str, Any],
                         next_state: Dict[str, Any]) -> float:
        """Get reward specific to patrol option."""
        reward = 0.0
        
        # Reward for covering patrol points
        if "patrol_points" in state:
            min_dist = min(self._get_distance(next_state["agent_position"], point)
                          for point in state["patrol_points"])
            reward -= min_dist * 0.1
            
        # Reward for discovering opponents
        if "opponent_positions" in state:
            min_dist = min(self._get_distance(next_state["agent_position"], opp_pos)
                          for opp_pos in state["opponent_positions"])
            if min_dist < 10.0:
                reward += 0.5
                
        return reward
        
    def _get_evade_reward(self, state: Dict[str, Any],
                        next_state: Dict[str, Any]) -> float:
        """Get reward specific to evade option."""
        reward = 0.0
        
        # Reward for increasing distance from opponents
        if "opponent_positions" in state:
            prev_min_dist = min(self._get_distance(state["agent_position"], opp_pos)
                              for opp_pos in state["opponent_positions"])
            next_min_dist = min(self._get_distance(next_state["agent_position"], opp_pos)
                              for opp_pos in state["opponent_positions"])
            reward += (next_min_dist - prev_min_dist) * 0.2
            
        # Penalty for being tagged
        if "is_tagged" in next_state and next_state["is_tagged"]:
            reward -= 2.0
            
        return reward
        
    def _get_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate distance between two positions."""
        return np.linalg.norm(pos1 - pos2) 