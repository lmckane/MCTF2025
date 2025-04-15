from typing import Dict, Any, List, Tuple
import numpy as np

class OptionAttention:
    """Implements attention mechanisms for options."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.attention_weights = {}
        self.attention_history = {}
        
    def compute_attention(self, option_name: str, state: Dict[str, Any],
                        features: List[str]) -> Dict[str, float]:
        """
        Compute attention weights for state features.
        
        Args:
            option_name: Name of the option
            state: Current state
            features: List of features to attend to
            
        Returns:
            Dict[str, float]: Dictionary of attention weights
        """
        if option_name not in self.attention_weights:
            self.attention_weights[option_name] = {
                feature: 1.0 / len(features) for feature in features
            }
            
        # Compute attention scores
        scores = {}
        for feature in features:
            if feature in state:
                # Base attention from weights
                score = self.attention_weights[option_name][feature]
                
                # Adjust based on feature importance
                importance = self._get_feature_importance(feature, state)
                score *= importance
                
                # Adjust based on feature value
                value = self._get_feature_value(feature, state)
                score *= value
                
                scores[feature] = score
                
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}
            
        # Update attention history
        self._update_attention_history(option_name, scores)
        
        return scores
        
    def _get_feature_importance(self, feature: str,
                              state: Dict[str, Any]) -> float:
        """Get importance score for a feature."""
        importance = 1.0
        
        # Adjust importance based on feature type
        if feature.endswith("_distance"):
            importance *= 2.0
        elif feature.endswith("_position"):
            importance *= 1.5
        elif feature == "has_flag":
            importance *= 3.0
            
        # Adjust based on feature value
        if feature in state:
            value = state[feature]
            if isinstance(value, (int, float)):
                if value < 0:
                    importance *= 1.2
                    
        return importance
        
    def _get_feature_value(self, feature: str,
                         state: Dict[str, Any]) -> float:
        """Get value score for a feature."""
        if feature not in state:
            return 0.0
            
        value = state[feature]
        
        if isinstance(value, (int, float)):
            # Normalize value to [0, 1]
            if feature.endswith("_distance"):
                max_dist = self.config.get("max_distance", 100.0)
                return 1.0 - (value / max_dist)
            else:
                return abs(value)
        elif isinstance(value, bool):
            return 1.0 if value else 0.0
        else:
            return 1.0
            
    def _update_attention_history(self, option_name: str,
                                attention_scores: Dict[str, float]):
        """Update attention history for an option."""
        if option_name not in self.attention_history:
            self.attention_history[option_name] = []
            
        self.attention_history[option_name].append(attention_scores)
        
        # Keep only recent history
        max_history = self.config.get("max_attention_history", 100)
        if len(self.attention_history[option_name]) > max_history:
            self.attention_history[option_name] = (
                self.attention_history[option_name][-max_history:]
            )
            
    def update_attention_weights(self, option_name: str,
                               feature: str, weight: float):
        """
        Update attention weight for a feature.
        
        Args:
            option_name: Name of the option
            feature: Name of the feature
            weight: New weight value
        """
        if option_name not in self.attention_weights:
            self.attention_weights[option_name] = {}
            
        self.attention_weights[option_name][feature] = weight
        
    def get_attention_statistics(self, option_name: str) -> Dict[str, Any]:
        """
        Get statistics about attention for an option.
        
        Args:
            option_name: Name of the option
            
        Returns:
            Dict[str, Any]: Dictionary of attention statistics
        """
        if option_name not in self.attention_history:
            return {}
            
        history = self.attention_history[option_name]
        
        if not history:
            return {}
            
        # Calculate statistics for each feature
        stats = {}
        features = set().union(*[h.keys() for h in history])
        
        for feature in features:
            weights = [h.get(feature, 0.0) for h in history]
            stats[feature] = {
                "mean": np.mean(weights),
                "std": np.std(weights),
                "max": np.max(weights),
                "min": np.min(weights)
            }
            
        return stats
        
    def get_attention_trends(self, option_name: str,
                           feature: str) -> List[float]:
        """
        Get attention trend for a feature.
        
        Args:
            option_name: Name of the option
            feature: Name of the feature
            
        Returns:
            List[float]: List of attention weights over time
        """
        if option_name not in self.attention_history:
            return []
            
        return [h.get(feature, 0.0) for h in self.attention_history[option_name]]
        
    def reset(self):
        """Reset attention state."""
        self.attention_weights = {}
        self.attention_history = {} 