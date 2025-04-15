from typing import Dict, Any, List
import numpy as np
from datetime import datetime

class OptionMonitor:
    """Monitors and records option execution performance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.execution_history = {}
        self.metrics = {}
        self.start_time = datetime.now()
        
    def record_execution(self, option_name: str, state: Dict[str, Any], action: np.ndarray):
        """
        Record an option execution step.
        
        Args:
            option_name: Name of the executed option
            state: State during execution
            action: Action taken
        """
        # Initialize history for option if needed
        if option_name not in self.execution_history:
            self.execution_history[option_name] = []
            self.metrics[option_name] = {
                "total_steps": 0,
                "successful_steps": 0,
                "total_reward": 0,
                "dangerous_steps": 0
            }
            
        # Record execution
        execution_record = {
            "timestamp": datetime.now(),
            "state": state,
            "action": action,
            "step_number": self.metrics[option_name]["total_steps"]
        }
        self.execution_history[option_name].append(execution_record)
        
        # Update metrics
        self._update_metrics(option_name, state, action)
        
    def _update_metrics(self, option_name: str, state: Dict[str, Any], action: np.ndarray):
        """Update execution metrics."""
        metrics = self.metrics[option_name]
        metrics["total_steps"] += 1
        
        # Check for success
        if self._is_successful(state, option_name):
            metrics["successful_steps"] += 1
            
        # Check for danger
        if self._is_dangerous(state):
            metrics["dangerous_steps"] += 1
            
        # Update reward
        reward = self._calculate_reward(state, action, option_name)
        metrics["total_reward"] += reward
        
    def _is_successful(self, state: Dict[str, Any], option_name: str) -> bool:
        """Check if execution step was successful."""
        if option_name == "capture":
            return state.get("has_flag", False)
        elif option_name == "defend":
            return not state.get("flag_captured", False)
        elif option_name == "patrol":
            return state.get("area_covered", 0) > 0.8
        return False
        
    def _is_dangerous(self, state: Dict[str, Any]) -> bool:
        """Check if execution step was dangerous."""
        # Check if agent was tagged
        if state.get("is_tagged", False):
            return True
            
        # Check if agent was close to opponents
        if "opponent_distances" in state:
            min_distance = min(state["opponent_distances"])
            if min_distance < self.config.get("danger_threshold", 0.5):
                return True
                
        return False
        
    def _calculate_reward(self, state: Dict[str, Any], action: np.ndarray, option_name: str) -> float:
        """Calculate reward for execution step."""
        reward = 0.0
        
        # Base reward for taking action
        reward += 0.1
        
        # Option-specific rewards
        if option_name == "capture":
            # Reward for moving towards opponent flag
            agent_pos = state["agent_position"]
            flag_pos = state["opponent_flag_position"]
            prev_dist = np.linalg.norm(agent_pos - flag_pos)
            new_pos = agent_pos + action
            new_dist = np.linalg.norm(new_pos - flag_pos)
            reward += (prev_dist - new_dist) * 0.5
            
        elif option_name == "defend":
            # Reward for staying near own flag
            agent_pos = state["agent_position"]
            flag_pos = state["team_flag_position"]
            distance = np.linalg.norm(agent_pos - flag_pos)
            reward += 1.0 / (1.0 + distance)
            
        elif option_name == "patrol":
            # Reward for covering new areas
            reward += state.get("area_covered", 0) * 0.1
            
        # Penalty for danger
        if self._is_dangerous(state):
            reward -= 0.5
            
        return reward
        
    def get_option_metrics(self, option_name: str) -> Dict[str, float]:
        """Get metrics for an option."""
        if option_name not in self.metrics:
            return {}
            
        metrics = self.metrics[option_name]
        total_steps = metrics["total_steps"]
        
        return {
            "total_steps": total_steps,
            "success_rate": metrics["successful_steps"] / total_steps if total_steps > 0 else 0,
            "average_reward": metrics["total_reward"] / total_steps if total_steps > 0 else 0,
            "danger_rate": metrics["dangerous_steps"] / total_steps if total_steps > 0 else 0,
            "execution_time": (datetime.now() - self.start_time).total_seconds()
        }
        
    def get_execution_history(self, option_name: str) -> List[Dict[str, Any]]:
        """Get execution history for an option."""
        return self.execution_history.get(option_name, [])
        
    def reset(self):
        """Reset the monitor's state."""
        self.execution_history = {}
        self.metrics = {}
        self.start_time = datetime.now() 