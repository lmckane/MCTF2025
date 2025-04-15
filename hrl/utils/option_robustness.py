from typing import Dict, Any, List, Tuple
import numpy as np
from collections import defaultdict

class OptionRobustness:
    """Ensures options work reliably in different conditions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.robustness_threshold = config.get("robustness_threshold", 0.8)
        self.performance_history = defaultdict(list)
        self.condition_history = defaultdict(list)
        self.failure_modes = defaultdict(list)
        
    def update_performance(self, option_name: str,
                         performance: float,
                         conditions: Dict[str, Any]):
        """
        Update performance history for an option.
        
        Args:
            option_name: Name of the option
            performance: Performance metric
            conditions: Dictionary of environmental conditions
        """
        self.performance_history[option_name].append(performance)
        self.condition_history[option_name].append(conditions)
        
    def check_robustness(self, option_name: str) -> Dict[str, Any]:
        """
        Check robustness of an option.
        
        Args:
            option_name: Name of the option
            
        Returns:
            Dict[str, Any]: Robustness information
        """
        if option_name not in self.performance_history:
            return {"robust": False, "reason": "No performance data"}
            
        performances = self.performance_history[option_name]
        conditions = self.condition_history[option_name]
        
        # Compute basic statistics
        mean_performance = np.mean(performances)
        std_performance = np.std(performances)
        min_performance = np.min(performances)
        
        # Check performance consistency
        is_consistent = std_performance / (mean_performance + 1e-6) < 0.2
        
        # Check condition coverage
        condition_coverage = self._compute_condition_coverage(conditions)
        
        # Check for failure modes
        failure_modes = self._analyze_failure_modes(
            option_name, performances, conditions
        )
        
        # Determine robustness
        is_robust = (
            mean_performance >= self.robustness_threshold and
            is_consistent and
            condition_coverage >= 0.7 and
            len(failure_modes) == 0
        )
        
        return {
            "robust": is_robust,
            "mean_performance": float(mean_performance),
            "std_performance": float(std_performance),
            "min_performance": float(min_performance),
            "is_consistent": is_consistent,
            "condition_coverage": float(condition_coverage),
            "failure_modes": failure_modes
        }
        
    def _compute_condition_coverage(self, conditions: List[Dict[str, Any]]) -> float:
        """Compute coverage of different conditions."""
        if not conditions:
            return 0.0
            
        # Get all possible condition values
        condition_values = defaultdict(set)
        for cond in conditions:
            for k, v in cond.items():
                condition_values[k].add(v)
                
        # Compute coverage
        total_values = sum(len(values) for values in condition_values.values())
        if total_values == 0:
            return 0.0
            
        return float(len(conditions)) / total_values
        
    def _analyze_failure_modes(self, option_name: str,
                             performances: List[float],
                             conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze failure modes of an option."""
        failures = []
        threshold = self.robustness_threshold
        
        for i, (perf, cond) in enumerate(zip(performances, conditions)):
            if perf < threshold:
                failure = {
                    "performance": float(perf),
                    "conditions": cond,
                    "index": i
                }
                failures.append(failure)
                self.failure_modes[option_name].append(failure)
                
        return failures
        
    def get_robustness_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about option robustness.
        
        Returns:
            Dict[str, Any]: Dictionary of robustness statistics
        """
        stats = {
            "num_options": len(self.performance_history),
            "robust_options": 0,
            "average_performance": 0.0,
            "total_failure_modes": sum(
                len(modes) for modes in self.failure_modes.values()
            )
        }
        
        # Compute statistics
        performances = []
        for option in self.performance_history:
            robustness = self.check_robustness(option)
            if robustness["robust"]:
                stats["robust_options"] += 1
            performances.extend(self.performance_history[option])
            
        if performances:
            stats["average_performance"] = float(np.mean(performances))
            
        return stats
        
    def suggest_improvements(self, option_name: str) -> List[Dict[str, Any]]:
        """
        Suggest improvements for an option.
        
        Args:
            option_name: Name of the option
            
        Returns:
            List[Dict[str, Any]]: List of improvement suggestions
        """
        if option_name not in self.performance_history:
            return []
            
        robustness = self.check_robustness(option_name)
        suggestions = []
        
        # Suggest based on performance
        if robustness["mean_performance"] < self.robustness_threshold:
            suggestions.append({
                "type": "performance",
                "description": "Improve overall performance",
                "current_value": robustness["mean_performance"],
                "target_value": self.robustness_threshold
            })
            
        # Suggest based on consistency
        if not robustness["is_consistent"]:
            suggestions.append({
                "type": "consistency",
                "description": "Reduce performance variance",
                "current_value": robustness["std_performance"],
                "target_value": robustness["mean_performance"] * 0.2
            })
            
        # Suggest based on condition coverage
        if robustness["condition_coverage"] < 0.7:
            suggestions.append({
                "type": "coverage",
                "description": "Increase condition coverage",
                "current_value": robustness["condition_coverage"],
                "target_value": 0.7
            })
            
        # Suggest based on failure modes
        for failure in robustness["failure_modes"]:
            suggestions.append({
                "type": "failure_mode",
                "description": f"Address failure in conditions: {failure['conditions']}",
                "performance": failure["performance"],
                "conditions": failure["conditions"]
            })
            
        return suggestions
        
    def reset(self):
        """Reset robustness state."""
        self.performance_history.clear()
        self.condition_history.clear()
        self.failure_modes.clear() 