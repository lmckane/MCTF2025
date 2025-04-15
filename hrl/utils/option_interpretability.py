from typing import Dict, Any, List, Tuple
import numpy as np
from collections import defaultdict

class OptionInterpretability:
    """Makes option behavior understandable."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.interpretability_threshold = config.get("interpretability_threshold", 0.8)
        self.behavior_history = defaultdict(list)
        self.feature_importance = defaultdict(dict)
        self.decision_patterns = defaultdict(list)
        
    def update_behavior(self, option_name: str,
                       state: Dict[str, Any],
                       action: Dict[str, Any],
                       explanation: Dict[str, Any]):
        """
        Update behavior history for an option.
        
        Args:
            option_name: Name of the option
            state: Current state
            action: Action taken
            explanation: Explanation of the decision
        """
        self.behavior_history[option_name].append({
            "state": state,
            "action": action,
            "explanation": explanation
        })
        
    def compute_interpretability(self, option_name: str) -> Dict[str, Any]:
        """
        Compute interpretability of an option.
        
        Args:
            option_name: Name of the option
            
        Returns:
            Dict[str, Any]: Interpretability information
        """
        if option_name not in self.behavior_history:
            return {"interpretable": False, "reason": "No behavior data"}
            
        behaviors = self.behavior_history[option_name]
        
        # Compute interpretability metrics
        consistency_score = self._compute_consistency(option_name)
        clarity_score = self._compute_clarity(option_name)
        feature_importance = self._compute_feature_importance(option_name)
        decision_patterns = self._identify_decision_patterns(option_name)
        
        # Compute overall interpretability score
        interpretability_score = np.mean([
            consistency_score,
            clarity_score,
            self._compute_feature_importance_score(feature_importance)
        ])
        
        # Check if interpretable
        is_interpretable = interpretability_score >= self.interpretability_threshold
        
        return {
            "interpretable": is_interpretable,
            "interpretability_score": float(interpretability_score),
            "consistency_score": float(consistency_score),
            "clarity_score": float(clarity_score),
            "feature_importance": feature_importance,
            "decision_patterns": decision_patterns
        }
        
    def _compute_consistency(self, option_name: str) -> float:
        """Compute consistency of option behavior."""
        if option_name not in self.behavior_history:
            return 0.0
            
        behaviors = self.behavior_history[option_name]
        
        # Group similar states
        state_groups = defaultdict(list)
        for behavior in behaviors:
            state_key = self._get_state_key(behavior["state"])
            state_groups[state_key].append(behavior["action"])
            
        # Compute consistency within groups
        consistency_scores = []
        for actions in state_groups.values():
            if len(actions) > 1:
                # Compute action similarity
                similarity = self._compute_action_similarity(actions)
                consistency_scores.append(similarity)
                
        if not consistency_scores:
            return 0.0
            
        return float(np.mean(consistency_scores))
        
    def _compute_clarity(self, option_name: str) -> float:
        """Compute clarity of option explanations."""
        if option_name not in self.behavior_history:
            return 0.0
            
        behaviors = self.behavior_history[option_name]
        
        # Compute explanation clarity scores
        clarity_scores = []
        for behavior in behaviors:
            explanation = behavior["explanation"]
            clarity = self._compute_explanation_clarity(explanation)
            clarity_scores.append(clarity)
            
        if not clarity_scores:
            return 0.0
            
        return float(np.mean(clarity_scores))
        
    def _compute_feature_importance(self, option_name: str) -> Dict[str, float]:
        """Compute importance of state features."""
        if option_name not in self.behavior_history:
            return {}
            
        behaviors = self.behavior_history[option_name]
        
        # Initialize feature importance
        feature_importance = defaultdict(list)
        
        # Collect feature importance from explanations
        for behavior in behaviors:
            explanation = behavior["explanation"]
            if "feature_importance" in explanation:
                for feature, importance in explanation["feature_importance"].items():
                    feature_importance[feature].append(importance)
                    
        # Compute mean importance for each feature
        mean_importance = {}
        for feature, importances in feature_importance.items():
            mean_importance[feature] = float(np.mean(importances))
            
        return mean_importance
        
    def _identify_decision_patterns(self, option_name: str) -> List[Dict[str, Any]]:
        """Identify patterns in option decisions."""
        if option_name not in self.behavior_history:
            return []
            
        behaviors = self.behavior_history[option_name]
        patterns = []
        
        # Group behaviors by similar states
        state_groups = defaultdict(list)
        for behavior in behaviors:
            state_key = self._get_state_key(behavior["state"])
            state_groups[state_key].append(behavior)
            
        # Identify patterns in each group
        for state_key, group_behaviors in state_groups.items():
            if len(group_behaviors) > 1:
                pattern = self._extract_decision_pattern(group_behaviors)
                if pattern:
                    patterns.append(pattern)
                    
        return patterns
        
    def _extract_decision_pattern(self, behaviors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract decision pattern from behaviors."""
        # Get common state features
        common_features = set(behaviors[0]["state"].keys())
        for behavior in behaviors[1:]:
            common_features &= set(behavior["state"].keys())
            
        # Get common action features
        common_actions = set(behaviors[0]["action"].keys())
        for behavior in behaviors[1:]:
            common_actions &= set(behavior["action"].keys())
            
        # Extract pattern
        pattern = {
            "state_features": list(common_features),
            "action_features": list(common_actions),
            "count": len(behaviors),
            "example": behaviors[0]
        }
        
        return pattern
        
    def _get_state_key(self, state: Dict[str, Any]) -> str:
        """Get a unique key for a state."""
        # Create a string representation of the state
        key_parts = []
        
        for k, v in sorted(state.items()):
            if isinstance(v, (int, float)):
                # Round to 2 decimal places for discretization
                v = round(v, 2)
            key_parts.append(f"{k}:{v}")
            
        return "|".join(key_parts)
        
    def _compute_action_similarity(self, actions: List[Dict[str, Any]]) -> float:
        """Compute similarity between actions."""
        if not actions:
            return 0.0
            
        # Convert actions to feature vectors
        features = []
        for action in actions:
            feature_vector = []
            for k, v in sorted(action.items()):
                if isinstance(v, (int, float)):
                    feature_vector.append(v)
            features.append(feature_vector)
            
        features = np.array(features)
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                similarity = np.dot(features[i], features[j]) / (
                    np.linalg.norm(features[i]) * np.linalg.norm(features[j])
                )
                similarities.append(similarity)
                
        if not similarities:
            return 0.0
            
        return float(np.mean(similarities))
        
    def _compute_explanation_clarity(self, explanation: Dict[str, Any]) -> float:
        """Compute clarity of an explanation."""
        clarity_scores = []
        
        # Check explanation completeness
        if "reasoning" in explanation:
            clarity_scores.append(1.0)
        else:
            clarity_scores.append(0.0)
            
        # Check feature importance
        if "feature_importance" in explanation:
            clarity_scores.append(1.0)
        else:
            clarity_scores.append(0.0)
            
        # Check decision confidence
        if "confidence" in explanation:
            clarity_scores.append(float(explanation["confidence"]))
        else:
            clarity_scores.append(0.0)
            
        return float(np.mean(clarity_scores))
        
    def _compute_feature_importance_score(self, feature_importance: Dict[str, float]) -> float:
        """Compute score based on feature importance."""
        if not feature_importance:
            return 0.0
            
        # Check if importance is well-distributed
        importance_values = list(feature_importance.values())
        if len(importance_values) < 2:
            return 0.0
            
        # Compute entropy of importance distribution
        importance_sum = sum(importance_values)
        if importance_sum == 0:
            return 0.0
            
        normalized_importance = np.array(importance_values) / importance_sum
        entropy = -np.sum(normalized_importance * np.log2(normalized_importance + 1e-10))
        max_entropy = np.log2(len(importance_values))
        
        return float(entropy / max_entropy)
        
    def get_interpretability_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about option interpretability.
        
        Returns:
            Dict[str, Any]: Dictionary of interpretability statistics
        """
        stats = {
            "num_options": len(self.behavior_history),
            "interpretable_options": 0,
            "average_interpretability": 0.0,
            "total_patterns": sum(
                len(patterns) for patterns in self.decision_patterns.values()
            )
        }
        
        # Compute statistics
        interpretability_scores = []
        for option in self.behavior_history:
            interpretability = self.compute_interpretability(option)
            interpretability_scores.append(interpretability["interpretability_score"])
            if interpretability["interpretable"]:
                stats["interpretable_options"] += 1
                
        if interpretability_scores:
            stats["average_interpretability"] = float(np.mean(interpretability_scores))
            
        return stats
        
    def reset(self):
        """Reset interpretability state."""
        self.behavior_history.clear()
        self.feature_importance.clear()
        self.decision_patterns.clear() 