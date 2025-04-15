from typing import Dict, Any, List
import numpy as np
from hrl.policies.base import BaseHierarchicalPolicy
from hrl.options.base import BaseOption

class HierarchicalPolicy(BaseHierarchicalPolicy):
    """Hierarchical policy that selects and executes options."""
    
    def __init__(self, meta_policy: 'MetaPolicy', option_policies: Dict[str, 'OptionPolicy']):
        super().__init__({})
        self.meta_policy = meta_policy
        self.option_policies = option_policies
        self.current_option = None
        self.option_history = []
        
    def select_option(self, state: Dict[str, Any]) -> str:
        """Select the next option to execute based on the current state."""
        # Get available options
        available_options = []
        for option_name, option_policy in self.option_policies.items():
            if option_policy.initiate(state):
                available_options.append(option_name)
                
        if not available_options:
            return None
            
        # Use meta-policy to select the best option
        option_scores = {}
        for option_name in available_options:
            score = self.meta_policy.get_option_score(state, option_name)
            option_scores[option_name] = score
            
        # Select option with highest score
        selected_option = max(option_scores.items(), key=lambda x: x[1])[0]
        self.current_option = selected_option
        self.option_history.append(selected_option)
        
        return selected_option
        
    def select_action(self, state: Dict[str, Any]) -> np.ndarray:
        """Select the action to take based on the current state and option."""
        if self.current_option is None:
            self.current_option = self.select_option(state)
            
        if self.current_option is None:
            return np.zeros(2)  # Default no-op action
            
        # Get action from current option's policy
        option_policy = self.option_policies[self.current_option]
        action = option_policy.get_action(state)
        
        # Check if current option should terminate
        if option_policy.terminate(state):
            self.current_option = None
            
        return action
        
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """Update the policy based on the transition."""
        # Update meta-policy
        self.meta_policy.update(state, action, reward, next_state, done)
        
        # Update current option's policy if active
        if self.current_option is not None:
            option_policy = self.option_policies[self.current_option]
            option_policy.update(state, action, reward, next_state, done)
            
        # Reset if episode is done
        if done:
            self.reset()
            
    def reset(self):
        """Reset the policy's internal state."""
        super().reset()
        self.meta_policy.reset()
        for option_policy in self.option_policies.values():
            option_policy.reset() 