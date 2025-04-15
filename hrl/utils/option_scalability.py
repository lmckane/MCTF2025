from typing import Dict, Any, List, Tuple
import numpy as np
from collections import defaultdict

class OptionScalability:
    """Manages system scalability."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scalability_threshold = config.get("scalability_threshold", 0.8)
        self.performance_history = defaultdict(list)
        self.scale_history = defaultdict(list)
        self.bottleneck_history = defaultdict(list)
        
    def update_performance(self, option_name: str,
                         performance: float,
                         scale_factors: Dict[str, float]):
        """
        Update performance at different scales.
        
        Args:
            option_name: Name of the option
            performance: Performance metric
            scale_factors: Dictionary of scale factors
        """
        self.performance_history[option_name].append(performance)
        self.scale_history[option_name].append(scale_factors)
        
    def check_scalability(self, option_name: str) -> Dict[str, Any]:
        """
        Check scalability of an option.
        
        Args:
            option_name: Name of the option
            
        Returns:
            Dict[str, Any]: Scalability information
        """
        if option_name not in self.performance_history:
            return {"scalable": False, "reason": "No performance data"}
            
        performances = self.performance_history[option_name]
        scale_factors = self.scale_history[option_name]
        
        # Compute scalability metrics
        scalability_score = self._compute_scalability_score(performances, scale_factors)
        bottlenecks = self._identify_bottlenecks(option_name)
        
        # Check if scalable
        is_scalable = scalability_score >= self.scalability_threshold
        
        return {
            "scalable": is_scalable,
            "scalability_score": scalability_score,
            "bottlenecks": bottlenecks
        }
        
    def _compute_scalability_score(self, performances: List[float],
                                 scale_factors: List[Dict[str, float]]) -> float:
        """Compute scalability score."""
        if not performances or not scale_factors:
            return 0.0
            
        # Get all scale dimensions
        scale_dimensions = set()
        for factors in scale_factors:
            scale_dimensions.update(factors.keys())
            
        # Compute performance degradation for each dimension
        degradation_scores = []
        for dimension in scale_dimensions:
            # Get scale values and corresponding performances
            scales = []
            perfs = []
            for i, factors in enumerate(scale_factors):
                if dimension in factors:
                    scales.append(factors[dimension])
                    perfs.append(performances[i])
                    
            if len(scales) > 1:
                # Compute degradation rate
                degradation = self._compute_degradation_rate(scales, perfs)
                degradation_scores.append(1.0 - degradation)
                
        if not degradation_scores:
            return 0.0
            
        return float(np.mean(degradation_scores))
        
    def _compute_degradation_rate(self, scales: List[float],
                                performances: List[float]) -> float:
        """Compute performance degradation rate."""
        # Normalize scales and performances
        scales = np.array(scales)
        performances = np.array(performances)
        
        scales = scales / scales[0]
        performances = performances / performances[0]
        
        # Compute degradation rate
        if len(scales) > 1:
            degradation = np.polyfit(scales, performances, 1)[0]
            return float(abs(degradation))
        return 0.0
        
    def _identify_bottlenecks(self, option_name: str) -> List[Dict[str, Any]]:
        """Identify scalability bottlenecks."""
        if option_name not in self.performance_history:
            return []
            
        performances = self.performance_history[option_name]
        scale_factors = self.scale_history[option_name]
        
        bottlenecks = []
        
        # Get all scale dimensions
        scale_dimensions = set()
        for factors in scale_factors:
            scale_dimensions.update(factors.keys())
            
        # Check each dimension
        for dimension in scale_dimensions:
            # Get scale values and corresponding performances
            scales = []
            perfs = []
            for i, factors in enumerate(scale_factors):
                if dimension in factors:
                    scales.append(factors[dimension])
                    perfs.append(performances[i])
                    
            if len(scales) > 1:
                # Compute degradation rate
                degradation = self._compute_degradation_rate(scales, perfs)
                
                # Check if bottleneck
                if degradation > 0.2:  # Threshold for bottleneck
                    bottlenecks.append({
                        "dimension": dimension,
                        "degradation_rate": degradation,
                        "scale_range": {
                            "min": float(min(scales)),
                            "max": float(max(scales))
                        },
                        "performance_range": {
                            "min": float(min(perfs)),
                            "max": float(max(perfs))
                        }
                    })
                    
        return bottlenecks
        
    def optimize_scalability(self, option_name: str) -> Dict[str, Any]:
        """
        Optimize scalability of an option.
        
        Args:
            option_name: Name of the option
            
        Returns:
            Dict[str, Any]: Optimization information
        """
        if option_name not in self.performance_history:
            return {"success": False, "reason": "No performance data"}
            
        # Get current scalability
        scalability = self.check_scalability(option_name)
        
        # Generate optimization suggestions
        suggestions = self._generate_scalability_suggestions(
            option_name, scalability
        )
        
        return {
            "success": True,
            "current_scalability": scalability,
            "suggestions": suggestions
        }
        
    def _generate_scalability_suggestions(self, option_name: str,
                                        scalability: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate scalability optimization suggestions."""
        suggestions = []
        
        # Check overall scalability
        if not scalability["scalable"]:
            suggestions.append({
                "type": "overall",
                "description": "Improve overall scalability",
                "current_score": scalability["scalability_score"],
                "target_score": self.scalability_threshold
            })
            
        # Address bottlenecks
        for bottleneck in scalability["bottlenecks"]:
            suggestions.append({
                "type": "bottleneck",
                "dimension": bottleneck["dimension"],
                "description": f"Address scalability bottleneck in {bottleneck['dimension']}",
                "degradation_rate": bottleneck["degradation_rate"],
                "scale_range": bottleneck["scale_range"]
            })
            
        return suggestions
        
    def get_scalability_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about option scalability.
        
        Returns:
            Dict[str, Any]: Dictionary of scalability statistics
        """
        stats = {
            "num_options": len(self.performance_history),
            "scalable_options": 0,
            "average_scalability": 0.0,
            "total_bottlenecks": sum(
                len(bottlenecks) for bottlenecks in self.bottleneck_history.values()
            )
        }
        
        # Compute statistics
        scalability_scores = []
        for option in self.performance_history:
            scalability = self.check_scalability(option)
            scalability_scores.append(scalability["scalability_score"])
            if scalability["scalable"]:
                stats["scalable_options"] += 1
                
        if scalability_scores:
            stats["average_scalability"] = float(np.mean(scalability_scores))
            
        return stats
        
    def reset(self):
        """Reset scalability state."""
        self.performance_history.clear()
        self.scale_history.clear()
        self.bottleneck_history.clear() 