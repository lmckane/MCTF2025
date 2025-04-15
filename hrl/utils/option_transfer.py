from typing import Dict, Any, List, Tuple
import numpy as np
from collections import defaultdict

class OptionTransfer:
    """Handles transferring options between different scenarios."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.transfer_memory = defaultdict(dict)
        self.scenario_features = {}
        self.option_mappings = {}
        
    def store_scenario_features(self, scenario_id: str,
                              features: Dict[str, Any]):
        """
        Store features for a scenario.
        
        Args:
            scenario_id: Identifier for the scenario
            features: Dictionary of scenario features
        """
        self.scenario_features[scenario_id] = features
        
    def compute_scenario_similarity(self, scenario1: str,
                                  scenario2: str) -> float:
        """
        Compute similarity between two scenarios.
        
        Args:
            scenario1: First scenario identifier
            scenario2: Second scenario identifier
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if scenario1 not in self.scenario_features or scenario2 not in self.scenario_features:
            return 0.0
            
        features1 = self.scenario_features[scenario1]
        features2 = self.scenario_features[scenario2]
        
        # Compute feature similarity
        similarity = self._compute_feature_similarity(features1, features2)
        return similarity
        
    def _compute_feature_similarity(self, features1: Dict[str, Any],
                                  features2: Dict[str, Any]) -> float:
        """Compute similarity between feature sets."""
        common_features = set(features1.keys()) & set(features2.keys())
        if not common_features:
            return 0.0
            
        similarities = []
        for feature in common_features:
            v1 = features1[feature]
            v2 = features2[feature]
            
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                # Numerical similarity
                similarity = 1.0 - abs(v1 - v2) / max(abs(v1), abs(v2), 1e-6)
            elif isinstance(v1, str) and isinstance(v2, str):
                # String similarity (simple equality)
                similarity = 1.0 if v1 == v2 else 0.0
            else:
                # Default similarity
                similarity = 0.0
                
            similarities.append(similarity)
            
        return float(np.mean(similarities))
        
    def transfer_option(self, source_scenario: str,
                       target_scenario: str,
                       option_name: str) -> Dict[str, Any]:
        """
        Transfer option from source to target scenario.
        
        Args:
            source_scenario: Source scenario identifier
            target_scenario: Target scenario identifier
            option_name: Name of option to transfer
            
        Returns:
            Dict[str, Any]: Transfer information
        """
        if source_scenario not in self.transfer_memory:
            return {"success": False, "reason": "Source scenario not found"}
            
        if option_name not in self.transfer_memory[source_scenario]:
            return {"success": False, "reason": "Option not found in source scenario"}
            
        # Get option information
        option_info = self.transfer_memory[source_scenario][option_name]
        
        # Compute scenario similarity
        similarity = self.compute_scenario_similarity(
            source_scenario, target_scenario
        )
        
        # Create transferred option
        transferred_option = self._adapt_option(
            option_info, source_scenario, target_scenario
        )
        
        return {
            "success": True,
            "similarity": similarity,
            "transferred_option": transferred_option
        }
        
    def _adapt_option(self, option_info: Dict[str, Any],
                     source_scenario: str,
                     target_scenario: str) -> Dict[str, Any]:
        """Adapt option for target scenario."""
        # Get scenario features
        source_features = self.scenario_features[source_scenario]
        target_features = self.scenario_features[target_scenario]
        
        # Create adapted option
        adapted_option = option_info.copy()
        
        # Adapt parameters based on scenario differences
        for param, value in option_info.get("parameters", {}).items():
            if param in source_features and param in target_features:
                # Scale parameter based on scenario difference
                scale_factor = target_features[param] / source_features[param]
                adapted_option["parameters"][param] = value * scale_factor
                
        return adapted_option
        
    def store_option(self, scenario_id: str,
                    option_name: str,
                    option_info: Dict[str, Any]):
        """
        Store option information for a scenario.
        
        Args:
            scenario_id: Scenario identifier
            option_name: Name of the option
            option_info: Dictionary of option information
        """
        self.transfer_memory[scenario_id][option_name] = option_info
        
    def get_transfer_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about option transfer.
        
        Returns:
            Dict[str, Any]: Dictionary of transfer statistics
        """
        stats = {
            "num_scenarios": len(self.scenario_features),
            "total_options": sum(
                len(options) for options in self.transfer_memory.values()
            ),
            "average_scenario_similarity": 0.0
        }
        
        # Compute average scenario similarity
        similarities = []
        scenarios = list(self.scenario_features.keys())
        for i in range(len(scenarios)):
            for j in range(i + 1, len(scenarios)):
                similarity = self.compute_scenario_similarity(
                    scenarios[i], scenarios[j]
                )
                similarities.append(similarity)
                
        if similarities:
            stats["average_scenario_similarity"] = float(np.mean(similarities))
            
        return stats
        
    def reset(self):
        """Reset transfer state."""
        self.transfer_memory.clear()
        self.scenario_features.clear()
        self.option_mappings.clear() 