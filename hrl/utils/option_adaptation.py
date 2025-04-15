from typing import Dict, Any, List
import numpy as np

class OptionAdapter:
    """Adapts options based on performance metrics and environmental changes."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adaptation_threshold = config.get("adaptation_threshold", 0.1)
        self.performance_history = {}
        self.adaptation_count = {}
        
    def adapt_option(self, option_name: str, performance_metrics: Dict[str, float],
                    state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt an option based on its performance and current state.
        
        Args:
            option_name: Name of the option to adapt
            performance_metrics: Dictionary of performance metrics
            state: Current environment state
            
        Returns:
            Dict[str, Any]: Adaptation parameters
        """
        if option_name not in self.performance_history:
            self.performance_history[option_name] = []
            self.adaptation_count[option_name] = 0
            
        # Update performance history
        self.performance_history[option_name].append(performance_metrics)
        
        # Check if adaptation is needed
        if self._should_adapt(option_name):
            adaptation = self._compute_adaptation(option_name, state)
            self.adaptation_count[option_name] += 1
            return adaptation
            
        return {}
        
    def _should_adapt(self, option_name: str) -> bool:
        """Determine if an option should be adapted."""
        if len(self.performance_history[option_name]) < 10:
            return False
            
        recent_performance = self.performance_history[option_name][-10:]
        avg_performance = np.mean([p["success_rate"] for p in recent_performance])
        
        return avg_performance < self.adaptation_threshold
        
    def _compute_adaptation(self, option_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Compute adaptation parameters for an option."""
        adaptation = {}
        
        # Adjust action parameters
        if "action_parameters" in self.config:
            adaptation["action_parameters"] = self._adapt_action_parameters(
                option_name, state
            )
            
        # Adjust termination conditions
        if "termination_conditions" in self.config:
            adaptation["termination_conditions"] = self._adapt_termination_conditions(
                option_name, state
            )
            
        # Adjust reward shaping
        if "reward_shaping" in self.config:
            adaptation["reward_shaping"] = self._adapt_reward_shaping(
                option_name, state
            )
            
        return adaptation
        
    def _adapt_action_parameters(self, option_name: str,
                               state: Dict[str, Any]) -> Dict[str, float]:
        """Adapt action parameters based on performance."""
        params = {}
        
        # Adjust speed based on success rate
        success_rate = np.mean([p["success_rate"] 
                              for p in self.performance_history[option_name][-5:]])
        params["speed_factor"] = 1.0 + (1.0 - success_rate) * 0.2
        
        # Adjust exploration rate
        params["exploration_rate"] = max(0.1, 1.0 - success_rate)
        
        return params
        
    def _adapt_termination_conditions(self, option_name: str,
                                    state: Dict[str, Any]) -> Dict[str, float]:
        """Adapt termination conditions based on performance."""
        conditions = {}
        
        # Adjust timeout based on average duration
        durations = [p["duration"] for p in self.performance_history[option_name][-5:]]
        avg_duration = np.mean(durations)
        conditions["timeout"] = avg_duration * 1.2
        
        # Adjust distance thresholds
        if "distance_threshold" in self.config:
            conditions["distance_threshold"] = self.config["distance_threshold"] * 0.9
            
        return conditions
        
    def _adapt_reward_shaping(self, option_name: str,
                            state: Dict[str, Any]) -> Dict[str, float]:
        """Adapt reward shaping parameters based on performance."""
        shaping = {}
        
        # Adjust reward weights based on success rate
        success_rate = np.mean([p["success_rate"] 
                              for p in self.performance_history[option_name][-5:]])
        
        if success_rate < 0.5:
            shaping["success_weight"] = 1.5
            shaping["failure_weight"] = 0.5
        else:
            shaping["success_weight"] = 1.0
            shaping["failure_weight"] = 1.0
            
        return shaping
        
    def reset(self):
        """Reset adaptation history."""
        self.performance_history = {}
        self.adaptation_count = {} 