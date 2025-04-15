from typing import Dict, Any, List, Tuple
import numpy as np
from hrl.utils.experience_buffer import ExperienceBuffer

class OptionLearner:
    """Learns and improves options through experience."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.learning_rate = config.get("learning_rate", 0.01)
        self.discount_factor = config.get("discount_factor", 0.99)
        self.batch_size = config.get("batch_size", 32)
        self.experience_buffer = ExperienceBuffer(config)
        
    def learn_from_experience(self, option_name: str, 
                            experiences: List[Tuple[Dict[str, Any], np.ndarray, float, Dict[str, Any], bool]]):
        """
        Learn from a batch of experiences.
        
        Args:
            option_name: Name of the option being learned
            experiences: List of (state, action, reward, next_state, done) tuples
        """
        # Add experiences to buffer
        for exp in experiences:
            self.experience_buffer.add(exp)
            
        # Sample batch and learn
        if len(self.experience_buffer) >= self.batch_size:
            batch = self.experience_buffer.sample(self.batch_size)
            self._update_option(batch, option_name)
            
    def _update_option(self, batch: List[Tuple], option_name: str):
        """Update option parameters based on batch of experiences."""
        for state, action, reward, next_state, done in batch:
            # Get current Q-value
            current_q = self._get_q_value(state, action, option_name)
            
            # Get target Q-value
            if done:
                target_q = reward
            else:
                next_q = self._get_max_q_value(next_state, option_name)
                target_q = reward + self.discount_factor * next_q
                
            # Update Q-value
            new_q = current_q + self.learning_rate * (target_q - current_q)
            self._set_q_value(state, action, new_q, option_name)
            
    def _get_q_value(self, state: Dict[str, Any], action: np.ndarray, option_name: str) -> float:
        """Get Q-value for state-action pair."""
        state_key = self._get_state_key(state)
        action_key = self._get_action_key(action)
        return self._get_option_values(option_name).get((state_key, action_key), 0.0)
        
    def _get_max_q_value(self, state: Dict[str, Any], option_name: str) -> float:
        """Get maximum Q-value for state."""
        state_key = self._get_state_key(state)
        option_values = self._get_option_values(option_name)
        max_q = float('-inf')
        
        for (s_key, a_key), q in option_values.items():
            if s_key == state_key and q > max_q:
                max_q = q
                
        return max_q if max_q != float('-inf') else 0.0
        
    def _set_q_value(self, state: Dict[str, Any], action: np.ndarray, value: float, option_name: str):
        """Set Q-value for state-action pair."""
        state_key = self._get_state_key(state)
        action_key = self._get_action_key(action)
        self._get_option_values(option_name)[(state_key, action_key)] = value
        
    def _get_state_key(self, state: Dict[str, Any]) -> str:
        """Convert state to string key."""
        return str(state)
        
    def _get_action_key(self, action: np.ndarray) -> str:
        """Convert action to string key."""
        return str(action.tolist())
        
    def _get_option_values(self, option_name: str) -> Dict[Tuple[str, str], float]:
        """Get Q-values for option."""
        if not hasattr(self, '_option_values'):
            self._option_values = {}
        if option_name not in self._option_values:
            self._option_values[option_name] = {}
        return self._option_values[option_name] 