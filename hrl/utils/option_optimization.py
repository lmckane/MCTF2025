from typing import Dict, Any, List
import numpy as np
from hrl.utils.option_evaluation import OptionEvaluator

class OptionOptimizer:
    """Optimizes option parameters through evaluation and improvement."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.evaluator = OptionEvaluator(config)
        self.optimization_steps = config.get("optimization_steps", 100)
        self.learning_rate = config.get("learning_rate", 0.01)
        
    def optimize_option(self, option_name: str, 
                       experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize an option's parameters.
        
        Args:
            option_name: Name of the option to optimize
            experiences: List of experiences to learn from
            
        Returns:
            Dict[str, Any]: Optimized option parameters
        """
        best_params = self._get_current_params(option_name)
        best_score = self.evaluator.evaluate_option(option_name, experiences)
        
        for _ in range(self.optimization_steps):
            # Generate candidate parameters
            candidate_params = self._generate_candidate_params(best_params)
            
            # Evaluate candidate
            self._set_option_params(option_name, candidate_params)
            candidate_score = self.evaluator.evaluate_option(option_name, experiences)
            
            # Update if better
            if candidate_score > best_score:
                best_params = candidate_params
                best_score = candidate_score
                
        # Set final parameters
        self._set_option_params(option_name, best_params)
        return best_params
        
    def _get_current_params(self, option_name: str) -> Dict[str, Any]:
        """Get current option parameters."""
        if not hasattr(self, '_option_params'):
            self._option_params = {}
        if option_name not in self._option_params:
            self._option_params[option_name] = self._get_default_params(option_name)
        return self._option_params[option_name]
        
    def _set_option_params(self, option_name: str, params: Dict[str, Any]):
        """Set option parameters."""
        if not hasattr(self, '_option_params'):
            self._option_params = {}
        self._option_params[option_name] = params
        
    def _generate_candidate_params(self, current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate new candidate parameters."""
        candidate = {}
        for key, value in current_params.items():
            if isinstance(value, (int, float)):
                # Add noise to numeric parameters
                noise = np.random.normal(0, self.learning_rate)
                candidate[key] = value + noise
            else:
                candidate[key] = value
        return candidate
        
    def _get_default_params(self, option_name: str) -> Dict[str, Any]:
        """Get default parameters for option type."""
        if option_name == "capture":
            return {
                "speed": 1.0,
                "aggression": 0.8,
                "caution": 0.2
            }
        elif option_name == "defend":
            return {
                "speed": 0.8,
                "aggression": 0.6,
                "caution": 0.4
            }
        elif option_name == "patrol":
            return {
                "speed": 0.6,
                "aggression": 0.4,
                "caution": 0.6
            }
        return {} 