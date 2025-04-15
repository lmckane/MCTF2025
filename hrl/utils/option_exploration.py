from typing import Dict, Any, List, Tuple
import numpy as np

class OptionExplorer:
    """Manages exploration strategies for options."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exploration_rate = config.get("exploration_rate", 0.1)
        self.min_exploration_rate = config.get("min_exploration_rate", 0.01)
        self.exploration_decay = config.get("exploration_decay", 0.995)
        self.option_counts = {}
        self.state_counts = {}
        
    def select_option(self, available_options: List[str],
                     option_scores: Dict[str, float]) -> str:
        """
        Select an option with exploration.
        
        Args:
            available_options: List of available options
            option_scores: Dictionary of option scores
            
        Returns:
            str: Selected option name
        """
        # Update exploration rate
        self._update_exploration_rate()
        
        # Get exploration probabilities
        exploration_probs = self._get_exploration_probs(
            available_options, option_scores
        )
        
        # Select option
        if np.random.random() < self.exploration_rate:
            # Exploration: select based on exploration probabilities
            return np.random.choice(
                available_options,
                p=list(exploration_probs.values())
            )
        else:
            # Exploitation: select best option
            return max(option_scores.items(), key=lambda x: x[1])[0]
            
    def _update_exploration_rate(self):
        """Update exploration rate with decay."""
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
        
    def _get_exploration_probs(self, available_options: List[str],
                             option_scores: Dict[str, float]) -> Dict[str, float]:
        """Get exploration probabilities for options."""
        # Initialize counts for new options
        for option in available_options:
            if option not in self.option_counts:
                self.option_counts[option] = 0
                
        # Calculate inverse visit counts
        total_counts = sum(self.option_counts[opt] for opt in available_options)
        if total_counts == 0:
            # Equal probability if no visits
            return {opt: 1.0 / len(available_options) for opt in available_options}
            
        # Calculate probabilities based on inverse visit counts
        probs = {}
        for option in available_options:
            count = self.option_counts[option]
            probs[option] = 1.0 / (1.0 + count)
            
        # Normalize probabilities
        total_prob = sum(probs.values())
        if total_prob > 0:
            probs = {k: v / total_prob for k, v in probs.items()}
            
        return probs
        
    def update_option_count(self, option_name: str):
        """Update visit count for an option."""
        if option_name not in self.option_counts:
            self.option_counts[option_name] = 0
            
        self.option_counts[option_name] += 1
        
    def update_state_count(self, state: Dict[str, Any]):
        """Update visit count for a state."""
        state_key = self._get_state_key(state)
        
        if state_key not in self.state_counts:
            self.state_counts[state_key] = 0
            
        self.state_counts[state_key] += 1
        
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
        
    def get_exploration_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about exploration.
        
        Returns:
            Dict[str, Any]: Dictionary of exploration statistics
        """
        stats = {
            "exploration_rate": self.exploration_rate,
            "unique_options": len(self.option_counts),
            "unique_states": len(self.state_counts),
            "average_option_visits": 0.0,
            "average_state_visits": 0.0
        }
        
        if stats["unique_options"] > 0:
            stats["average_option_visits"] = (
                sum(self.option_counts.values()) / stats["unique_options"]
            )
            
        if stats["unique_states"] > 0:
            stats["average_state_visits"] = (
                sum(self.state_counts.values()) / stats["unique_states"]
            )
            
        return stats
        
    def get_option_exploration_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get exploration statistics for each option.
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of option statistics
        """
        stats = {}
        
        for option, count in self.option_counts.items():
            stats[option] = {
                "visit_count": count,
                "exploration_prob": 1.0 / (1.0 + count)
            }
            
        return stats
        
    def reset(self):
        """Reset exploration state."""
        self.exploration_rate = self.config.get("exploration_rate", 0.1)
        self.option_counts = {}
        self.state_counts = {} 