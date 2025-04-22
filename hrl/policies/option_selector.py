import numpy as np
from typing import Dict, Any, List, Optional
from hrl.options.base import BaseOption
from hrl.utils.state_processor import StateProcessor

class OptionSelector:
    """Selects appropriate options based on agent state and game context."""
    
    def __init__(self, options: List[BaseOption], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the option selector.
        
        Args:
            options: List of available options
            config: Configuration dictionary with optional parameters:
                - option_weights: Initial weights for each option
                - learning_rate: Learning rate for option weights
                - min_option_duration: Minimum steps before switching options
                - max_option_duration: Maximum steps before forcing option switch
                - success_threshold: Threshold for considering an option successful
        """
        self.options = {opt.name: opt for opt in options}
        self.config = config or {}
        self.option_weights = self.config.get('option_weights', {opt.name: 1.0 for opt in options})
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.min_option_duration = self.config.get('min_option_duration', 10)
        self.max_option_duration = self.config.get('max_option_duration', 100)
        self.success_threshold = self.config.get('success_threshold', 0.7)
        
        self.current_option = None
        self.option_duration = 0
        self.option_history = []
        self.option_success_rates = {opt.name: 0.5 for opt in options}
        
    def select_option(self, state: Dict[str, Any]) -> BaseOption:
        """
        Select the most appropriate option based on current state.
        
        Args:
            state: Current environment state
            
        Returns:
            BaseOption: Selected option
        """
        # Check if current option should terminate
        if self.current_option is not None:
            if not self.current_option.terminate(state):
                self.option_duration += 1
                # Force switch if option has been running too long
                if self.option_duration >= self.max_option_duration:
                    self.current_option = None
                else:
                    return self.current_option
                    
        # Calculate option scores based on state
        option_scores = self._calculate_option_scores(state)
        
        # Apply option weights
        weighted_scores = {
            name: score * self.option_weights[name] * self.option_success_rates[name]
            for name, score in option_scores.items()
        }
        
        # Select option with highest weighted score
        selected_name = max(weighted_scores.items(), key=lambda x: x[1])[0]
        self.current_option = self.options[selected_name]
        self.option_duration = 0
        
        # Record selection
        self.option_history.append({
            'option': selected_name,
            'state': state,
            'score': weighted_scores[selected_name]
        })
        
        return self.current_option
        
    def _calculate_option_scores(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate scores for each option based on current state."""
        scores = {}
        
        # Check if agent has flag
        has_flag = state.get('agent_has_flag', False)
        is_tagged = state.get('agent_is_tagged', False)
        agent_health = state.get('agent_health', 1.0)
        
        # Calculate distances
        agent_pos = np.array(state.get('agent_position', [0, 0]))
        base_pos = np.array(state.get('base_position', [0, 0]))
        flag_pos = np.array(state.get('flag_position', [0, 0]))
        opponent_positions = [np.array(pos) for pos in state.get('opponent_positions', [])]
        
        # Calculate distances
        distance_to_base = np.linalg.norm(base_pos - agent_pos) if base_pos is not None else float('inf')
        distance_to_flag = np.linalg.norm(flag_pos - agent_pos) if flag_pos is not None else float('inf')
        nearest_opponent_distance = min(
            [np.linalg.norm(opp_pos - agent_pos) for opp_pos in opponent_positions],
            default=float('inf')
        )
        
        # Score each option
        for name, option in self.options.items():
            if name == 'attack_flag':
                # Prioritize attack if flag is close and we don't have it
                if not has_flag and distance_to_flag < 30:
                    scores[name] = 1.0 - (distance_to_flag / 30)
                else:
                    scores[name] = 0.1
                    
            elif name == 'capture_flag':
                # Prioritize capture if we have the flag
                if has_flag:
                    scores[name] = 1.0 - (distance_to_base / 100)
                else:
                    scores[name] = 0.0
                    
            elif name == 'guard_flag':
                # Prioritize guard if we're near our flag and opponents are close
                if distance_to_flag < 20 and nearest_opponent_distance < 30:
                    scores[name] = 1.0 - (nearest_opponent_distance / 30)
                else:
                    scores[name] = 0.2
                    
            elif name == 'evade':
                # Prioritize evade if opponents are too close
                if nearest_opponent_distance < 15:
                    scores[name] = 1.0 - (nearest_opponent_distance / 15)
                else:
                    scores[name] = 0.1
                    
            elif name == 'tag':
                # Prioritize tag if opponents are in our territory
                if nearest_opponent_distance < 25:
                    scores[name] = 1.0 - (nearest_opponent_distance / 25)
                else:
                    scores[name] = 0.1
                    
            elif name == 'retreat':
                # Prioritize retreat if we're tagged or low on health
                if is_tagged or agent_health < 0.3:
                    scores[name] = 1.0
                else:
                    scores[name] = 0.1
                    
            else:
                scores[name] = 0.1
                
        return scores
        
    def update_option_weights(self, reward: float, next_state: Dict[str, Any]):
        """
        Update option weights based on received reward.
        
        Args:
            reward: Reward received from the environment
            next_state: Next environment state
        """
        if self.current_option is None:
            return
            
        # Calculate success rate for current option
        success = reward > 0
        self.option_success_rates[self.current_option.name] = (
            0.9 * self.option_success_rates[self.current_option.name] +
            0.1 * float(success)
        )
        
        # Update option weights
        self.option_weights[self.current_option.name] += (
            self.learning_rate * reward * self.option_success_rates[self.current_option.name]
        )
        
        # Normalize weights
        total_weight = sum(self.option_weights.values())
        if total_weight > 0:
            self.option_weights = {
                name: weight / total_weight
                for name, weight in self.option_weights.items()
            }
            
    def get_option_history(self) -> List[Dict[str, Any]]:
        """Get the history of option selections."""
        return self.option_history
        
    def get_option_statistics(self) -> Dict[str, Any]:
        """Get statistics about option performance."""
        return {
            'weights': self.option_weights,
            'success_rates': self.option_success_rates,
            'total_selections': len(self.option_history)
        }
        
    def reset(self):
        """Reset the option selector's internal state."""
        self.current_option = None
        self.option_duration = 0
        self.option_history = []
        for opt in self.options.values():
            opt.reset() 