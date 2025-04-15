from typing import Dict, Any, Tuple
import numpy as np
from hrl.utils.option_monitoring import OptionMonitor

class OptionExecutor:
    """Executes options and manages their execution state."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitor = OptionMonitor(config)
        self.current_option = None
        self.option_state = {}
        
    def execute_option(self, option_name: str, state: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute an option in the current state.
        
        Args:
            option_name: Name of the option to execute
            state: Current environment state
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Action to take and updated option state
        """
        # Initialize option if new
        if option_name != self.current_option:
            self._initialize_option(option_name, state)
            
        # Get action from option
        action = self._get_option_action(option_name, state)
        
        # Update option state
        self._update_option_state(option_name, state, action)
        
        # Monitor execution
        self.monitor.record_execution(option_name, state, action)
        
        return action, self.option_state[option_name]
        
    def _initialize_option(self, option_name: str, state: Dict[str, Any]):
        """Initialize a new option."""
        self.current_option = option_name
        self.option_state[option_name] = {
            "steps": 0,
            "total_reward": 0,
            "success": False,
            "last_state": state
        }
        
    def _get_option_action(self, option_name: str, state: Dict[str, Any]) -> np.ndarray:
        """Get action from option based on current state."""
        if option_name == "capture":
            return self._get_capture_action(state)
        elif option_name == "defend":
            return self._get_defend_action(state)
        elif option_name == "patrol":
            return self._get_patrol_action(state)
        return np.zeros(2)
        
    def _update_option_state(self, option_name: str, state: Dict[str, Any], action: np.ndarray):
        """Update option's internal state."""
        option_state = self.option_state[option_name]
        option_state["steps"] += 1
        option_state["last_state"] = state
        
        # Update success status
        if option_name == "capture":
            option_state["success"] = state.get("has_flag", False)
        elif option_name == "defend":
            option_state["success"] = not state.get("flag_captured", False)
        elif option_name == "patrol":
            option_state["success"] = state.get("area_covered", 0) > 0.8
            
    def _get_capture_action(self, state: Dict[str, Any]) -> np.ndarray:
        """Get action for capture option."""
        agent_pos = state["agent_position"]
        flag_pos = state["opponent_flag_position"]
        
        # Move towards flag
        direction = flag_pos - agent_pos
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction = direction / distance
            
        # Add some noise for exploration
        noise = np.random.normal(0, 0.1, 2)
        return direction + noise
        
    def _get_defend_action(self, state: Dict[str, Any]) -> np.ndarray:
        """Get action for defend option."""
        agent_pos = state["agent_position"]
        flag_pos = state["team_flag_position"]
        
        # Find closest opponent to flag
        min_dist = float('inf')
        target_pos = flag_pos
        
        for opp_pos in state["opponent_positions"]:
            dist = np.linalg.norm(flag_pos - opp_pos)
            if dist < min_dist:
                min_dist = dist
                target_pos = opp_pos
                
        # Move towards target
        direction = target_pos - agent_pos
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction = direction / distance
            
        # Add some noise for exploration
        noise = np.random.normal(0, 0.1, 2)
        return direction + noise
        
    def _get_patrol_action(self, state: Dict[str, Any]) -> np.ndarray:
        """Get action for patrol option."""
        agent_pos = state["agent_position"]
        
        # Get next patrol point
        patrol_point = self._get_next_patrol_point(state)
        
        # Move towards patrol point
        direction = patrol_point - agent_pos
        distance = np.linalg.norm(direction)
        if distance > 0:
            direction = direction / distance
            
        # Add some noise for exploration
        noise = np.random.normal(0, 0.2, 2)
        return direction + noise
        
    def _get_next_patrol_point(self, state: Dict[str, Any]) -> np.ndarray:
        """Get next point to patrol to."""
        # Simple patrol pattern: move in a circle around the center
        center = np.array([0, 0])
        radius = 5.0
        angle_step = 0.1
        
        # Get current angle
        agent_pos = state["agent_position"]
        rel_pos = agent_pos - center
        current_angle = np.arctan2(rel_pos[1], rel_pos[0])
        
        # Calculate next point
        next_angle = current_angle + angle_step
        next_x = center[0] + radius * np.cos(next_angle)
        next_y = center[1] + radius * np.sin(next_angle)
        
        return np.array([next_x, next_y])
        
    def reset(self):
        """Reset the executor's state."""
        self.current_option = None
        self.option_state = {}
        self.monitor.reset() 