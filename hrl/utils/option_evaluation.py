from typing import Dict, Any, List
import numpy as np

class OptionEvaluator:
    """Evaluates option performance and quality."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.success_threshold = config.get("success_threshold", 0.8)
        
    def evaluate_option(self, option_name: str, 
                       experiences: List[Dict[str, Any]]) -> float:
        """
        Evaluate an option's performance.
        
        Args:
            option_name: Name of the option to evaluate
            experiences: List of experiences to evaluate from
            
        Returns:
            float: Evaluation score between 0 and 1
        """
        if not experiences:
            return 0.0
            
        # Calculate success rate
        successes = sum(1 for exp in experiences if self._is_successful(exp, option_name))
        success_rate = successes / len(experiences)
        
        # Calculate efficiency
        efficiency = self._calculate_efficiency(experiences, option_name)
        
        # Calculate safety
        safety = self._calculate_safety(experiences, option_name)
        
        # Combine metrics
        score = 0.4 * success_rate + 0.3 * efficiency + 0.3 * safety
        return score
        
    def _is_successful(self, experience: Dict[str, Any], option_name: str) -> bool:
        """Check if an experience was successful."""
        if option_name == "capture":
            return experience.get("has_flag", False)
        elif option_name == "defend":
            return not experience.get("flag_captured", False)
        elif option_name == "patrol":
            return experience.get("area_covered", 0) > self.success_threshold
        return False
        
    def _calculate_efficiency(self, experiences: List[Dict[str, Any]], 
                            option_name: str) -> float:
        """Calculate how efficiently the option achieves its goal."""
        if not experiences:
            return 0.0
            
        total_steps = sum(exp.get("steps", 0) for exp in experiences)
        total_reward = sum(exp.get("reward", 0) for exp in experiences)
        
        if total_steps == 0:
            return 0.0
            
        # Normalize efficiency score
        max_possible_reward = self._get_max_possible_reward(option_name)
        efficiency = total_reward / (total_steps * max_possible_reward)
        return max(0.0, min(1.0, efficiency))
        
    def _calculate_safety(self, experiences: List[Dict[str, Any]], 
                         option_name: str) -> float:
        """Calculate how safely the option executes."""
        if not experiences:
            return 0.0
            
        # Count dangerous situations
        dangers = sum(1 for exp in experiences if self._is_dangerous(exp))
        safety = 1.0 - (dangers / len(experiences))
        return max(0.0, min(1.0, safety))
        
    def _is_dangerous(self, experience: Dict[str, Any]) -> bool:
        """Check if an experience involved dangerous situations."""
        # Check if agent was tagged
        if experience.get("is_tagged", False):
            return True
            
        # Check if agent was close to opponents
        if "opponent_distances" in experience:
            min_distance = min(experience["opponent_distances"])
            if min_distance < self.config.get("danger_threshold", 0.5):
                return True
                
        return False
        
    def _get_max_possible_reward(self, option_name: str) -> float:
        """Get maximum possible reward for option type."""
        if option_name == "capture":
            return 100.0
        elif option_name == "defend":
            return 50.0
        elif option_name == "patrol":
            return 30.0
        return 0.0 