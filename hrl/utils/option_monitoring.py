from typing import Dict, Any, List
import numpy as np
from collections import defaultdict

class OptionMonitor:
    """Monitors option execution and collects statistics."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the option monitor."""
        self.config = config
        self.reset()
        
    def reset(self):
        """Reset monitoring data."""
        self.option_counts = defaultdict(int)
        self.option_durations = defaultdict(list)
        self.option_rewards = defaultdict(list)
        self.option_successes = defaultdict(int)
        self.option_activations = defaultdict(list)
        self.current_option = None
        self.current_option_steps = 0
        
    def record_execution(self, option_name: str, state: Dict[str, Any], action: np.ndarray):
        """Record option execution statistics."""
        # If this is a new option, update statistics for previous option
        if option_name != self.current_option and self.current_option is not None:
            self.option_durations[self.current_option].append(self.current_option_steps)
            
        # Update current option
        if option_name != self.current_option:
            self.option_counts[option_name] += 1
            self.current_option = option_name
            self.current_option_steps = 0
            
            # Record activation time
            step_count = state.get("step_count", 0)
            self.option_activations[option_name].append(step_count)
        
        # Increment step count for current option
        self.current_option_steps += 1
    
    def record_reward(self, option_name: str, reward: float):
        """Record reward received during option execution."""
        self.option_rewards[option_name].append(reward)
    
    def record_success(self, option_name: str, success: bool):
        """Record option success or failure."""
        if success:
            self.option_successes[option_name] += 1
    
    def get_option_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all options."""
        stats = {}
        
        for option_name in self.option_counts.keys():
            stats[option_name] = {
                "count": self.option_counts[option_name],
                "avg_duration": np.mean(self.option_durations[option_name]) if self.option_durations[option_name] else 0,
                "avg_reward": np.mean(self.option_rewards[option_name]) if self.option_rewards[option_name] else 0,
                "success_rate": self.option_successes[option_name] / max(1, self.option_counts[option_name])
            }
            
        return stats
    
    def get_option_activation_pattern(self) -> Dict[str, List[int]]:
        """Get option activation patterns over time."""
        return dict(self.option_activations) 