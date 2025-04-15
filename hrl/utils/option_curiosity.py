from typing import Dict, Any, List, Tuple
import numpy as np

class OptionCuriosity:
    """Adds curiosity-driven exploration to options."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.curiosity_factor = config.get("curiosity_factor", 0.1)
        self.exploration_bonus = config.get("exploration_bonus", 0.5)
        self.state_counts = {}
        self.option_counts = {}
        
    def get_curiosity_bonus(self, option_name: str, state: Dict[str, Any],
                          next_state: Dict[str, Any]) -> float:
        """
        Get curiosity bonus for a state transition.
        
        Args:
            option_name: Name of the option
            state: Current state
            next_state: Next state
            
        Returns:
            float: Curiosity bonus
        """
        # Get state novelty
        state_novelty = self._get_state_novelty(state)
        
        # Get transition novelty
        transition_novelty = self._get_transition_novelty(
            option_name, state, next_state
        )
        
        # Combine novelty scores
        novelty = (state_novelty + transition_novelty) / 2.0
        
        # Calculate curiosity bonus
        bonus = self.exploration_bonus * novelty * self.curiosity_factor
        
        # Update counts
        self._update_counts(option_name, state, next_state)
        
        return bonus
        
    def _get_state_novelty(self, state: Dict[str, Any]) -> float:
        """Get novelty score for a state."""
        state_key = self._get_state_key(state)
        
        if state_key not in self.state_counts:
            self.state_counts[state_key] = 0
            
        count = self.state_counts[state_key]
        novelty = 1.0 / (1.0 + count)
        
        return novelty
        
    def _get_transition_novelty(self, option_name: str,
                              state: Dict[str, Any],
                              next_state: Dict[str, Any]) -> float:
        """Get novelty score for a state transition."""
        transition_key = self._get_transition_key(option_name, state, next_state)
        
        if transition_key not in self.option_counts:
            self.option_counts[transition_key] = 0
            
        count = self.option_counts[transition_key]
        novelty = 1.0 / (1.0 + count)
        
        return novelty
        
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
        
    def _get_transition_key(self, option_name: str,
                          state: Dict[str, Any],
                          next_state: Dict[str, Any]) -> str:
        """Get a unique key for a state transition."""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        return f"{option_name}|{state_key}|{next_state_key}"
        
    def _update_counts(self, option_name: str,
                     state: Dict[str, Any],
                     next_state: Dict[str, Any]):
        """Update state and transition counts."""
        # Update state count
        state_key = self._get_state_key(state)
        if state_key not in self.state_counts:
            self.state_counts[state_key] = 0
        self.state_counts[state_key] += 1
        
        # Update transition count
        transition_key = self._get_transition_key(option_name, state, next_state)
        if transition_key not in self.option_counts:
            self.option_counts[transition_key] = 0
        self.option_counts[transition_key] += 1
        
    def get_exploration_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about exploration.
        
        Returns:
            Dict[str, Any]: Dictionary of exploration statistics
        """
        stats = {
            "unique_states": len(self.state_counts),
            "unique_transitions": len(self.option_counts),
            "average_state_visits": 0.0,
            "average_transition_visits": 0.0
        }
        
        if stats["unique_states"] > 0:
            stats["average_state_visits"] = (
                sum(self.state_counts.values()) / stats["unique_states"]
            )
            
        if stats["unique_transitions"] > 0:
            stats["average_transition_visits"] = (
                sum(self.option_counts.values()) / stats["unique_transitions"]
            )
            
        return stats
        
    def get_state_novelty_distribution(self) -> Dict[str, float]:
        """
        Get distribution of state novelty scores.
        
        Returns:
            Dict[str, float]: Dictionary of novelty statistics
        """
        if not self.state_counts:
            return {}
            
        novelties = [1.0 / (1.0 + count) for count in self.state_counts.values()]
        
        return {
            "mean": np.mean(novelties),
            "std": np.std(novelties),
            "min": np.min(novelties),
            "max": np.max(novelties)
        }
        
    def reset(self):
        """Reset exploration state."""
        self.state_counts = {}
        self.option_counts = {} 