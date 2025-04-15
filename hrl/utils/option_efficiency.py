from typing import Dict, Any, List, Tuple
import numpy as np
from collections import defaultdict

class OptionEfficiency:
    """Optimizes option performance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.efficiency_threshold = config.get("efficiency_threshold", 0.8)
        self.performance_history = defaultdict(list)
        self.resource_history = defaultdict(list)
        self.optimization_history = defaultdict(list)
        
    def update_performance(self, option_name: str,
                         performance: float,
                         resources: Dict[str, float]):
        """
        Update performance and resource usage for an option.
        
        Args:
            option_name: Name of the option
            performance: Performance metric
            resources: Dictionary of resource usage
        """
        self.performance_history[option_name].append(performance)
        self.resource_history[option_name].append(resources)
        
    def check_efficiency(self, option_name: str) -> Dict[str, Any]:
        """
        Check efficiency of an option.
        
        Args:
            option_name: Name of the option
            
        Returns:
            Dict[str, Any]: Efficiency information
        """
        if option_name not in self.performance_history:
            return {"efficient": False, "reason": "No performance data"}
            
        performances = self.performance_history[option_name]
        resources = self.resource_history[option_name]
        
        # Compute basic statistics
        mean_performance = np.mean(performances)
        mean_resources = self._compute_mean_resources(resources)
        efficiency_score = self._compute_efficiency_score(
            mean_performance, mean_resources
        )
        
        # Check if efficient
        is_efficient = efficiency_score >= self.efficiency_threshold
        
        return {
            "efficient": is_efficient,
            "efficiency_score": efficiency_score,
            "mean_performance": float(mean_performance),
            "mean_resources": mean_resources
        }
        
    def _compute_mean_resources(self, resources: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute mean resource usage."""
        if not resources:
            return {}
            
        # Initialize resource sums
        resource_sums = defaultdict(float)
        resource_counts = defaultdict(int)
        
        # Sum resources
        for resource_dict in resources:
            for resource, value in resource_dict.items():
                resource_sums[resource] += value
                resource_counts[resource] += 1
                
        # Compute means
        mean_resources = {}
        for resource, total in resource_sums.items():
            mean_resources[resource] = total / resource_counts[resource]
            
        return mean_resources
        
    def _compute_efficiency_score(self, performance: float,
                                resources: Dict[str, float]) -> float:
        """Compute efficiency score."""
        if not resources:
            return performance
            
        # Normalize resource usage
        normalized_resources = []
        for resource, value in resources.items():
            # Get resource weight from config
            weight = self.config.get(f"resource_weight_{resource}", 1.0)
            normalized_resources.append(value * weight)
            
        # Compute efficiency score
        resource_score = np.mean(normalized_resources)
        efficiency_score = performance / (1.0 + resource_score)
        
        return float(efficiency_score)
        
    def optimize_option(self, option_name: str) -> Dict[str, Any]:
        """
        Optimize an option's performance.
        
        Args:
            option_name: Name of the option
            
        Returns:
            Dict[str, Any]: Optimization information
        """
        if option_name not in self.performance_history:
            return {"success": False, "reason": "No performance data"}
            
        # Get current efficiency
        efficiency = self.check_efficiency(option_name)
        
        # Analyze resource usage
        resource_analysis = self._analyze_resources(option_name)
        
        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(
            option_name, efficiency, resource_analysis
        )
        
        # Update optimization history
        self.optimization_history[option_name].append({
            "efficiency": efficiency,
            "resource_analysis": resource_analysis,
            "suggestions": suggestions
        })
        
        return {
            "success": True,
            "current_efficiency": efficiency,
            "resource_analysis": resource_analysis,
            "suggestions": suggestions
        }
        
    def _analyze_resources(self, option_name: str) -> Dict[str, Any]:
        """Analyze resource usage patterns."""
        if option_name not in self.resource_history:
            return {}
            
        resources = self.resource_history[option_name]
        analysis = {}
        
        # Get all resource types
        resource_types = set()
        for resource_dict in resources:
            resource_types.update(resource_dict.keys())
            
        # Analyze each resource
        for resource in resource_types:
            values = [
                r.get(resource, 0.0) for r in resources
                if resource in r
            ]
            
            if values:
                analysis[resource] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
                
        return analysis
        
    def _generate_optimization_suggestions(self, option_name: str,
                                        efficiency: Dict[str, Any],
                                        resource_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization suggestions."""
        suggestions = []
        
        # Check performance
        if efficiency["mean_performance"] < self.efficiency_threshold:
            suggestions.append({
                "type": "performance",
                "description": "Improve overall performance",
                "current_value": efficiency["mean_performance"],
                "target_value": self.efficiency_threshold
            })
            
        # Check resource usage
        for resource, analysis in resource_analysis.items():
            if analysis["mean"] > self.config.get(f"resource_threshold_{resource}", 1.0):
                suggestions.append({
                    "type": "resource",
                    "resource": resource,
                    "description": f"Reduce {resource} usage",
                    "current_value": analysis["mean"],
                    "target_value": self.config.get(f"resource_threshold_{resource}", 1.0)
                })
                
        return suggestions
        
    def get_efficiency_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about option efficiency.
        
        Returns:
            Dict[str, Any]: Dictionary of efficiency statistics
        """
        stats = {
            "num_options": len(self.performance_history),
            "efficient_options": 0,
            "average_efficiency": 0.0,
            "total_optimizations": sum(
                len(history) for history in self.optimization_history.values()
            )
        }
        
        # Compute statistics
        efficiency_scores = []
        for option in self.performance_history:
            efficiency = self.check_efficiency(option)
            efficiency_scores.append(efficiency["efficiency_score"])
            if efficiency["efficient"]:
                stats["efficient_options"] += 1
                
        if efficiency_scores:
            stats["average_efficiency"] = float(np.mean(efficiency_scores))
            
        return stats
        
    def reset(self):
        """Reset efficiency state."""
        self.performance_history.clear()
        self.resource_history.clear()
        self.optimization_history.clear() 