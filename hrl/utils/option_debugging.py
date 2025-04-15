from typing import Dict, Any, List, Tuple
import numpy as np
from collections import defaultdict

class OptionDebugger:
    """Helps debug option execution."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.debug_threshold = config.get("debug_threshold", 0.5)
        self.execution_history = defaultdict(list)
        self.error_history = defaultdict(list)
        self.debug_points = defaultdict(list)
        
    def add_debug_point(self, option_name: str,
                       debug_point: Dict[str, Any]):
        """
        Add a debug point for an option.
        
        Args:
            option_name: Name of the option
            debug_point: Dictionary containing debug information
        """
        self.debug_points[option_name].append(debug_point)
        
    def log_execution(self, option_name: str,
                     state: Dict[str, Any],
                     action: Dict[str, Any],
                     result: Dict[str, Any]):
        """
        Log option execution.
        
        Args:
            option_name: Name of the option
            state: Current state
            action: Action taken
            result: Execution result
        """
        self.execution_history[option_name].append({
            "state": state,
            "action": action,
            "result": result
        })
        
        # Check for errors
        if "error" in result:
            self.error_history[option_name].append({
                "state": state,
                "action": action,
                "error": result["error"]
            })
            
    def analyze_execution(self, option_name: str) -> Dict[str, Any]:
        """
        Analyze option execution.
        
        Args:
            option_name: Name of the option
            
        Returns:
            Dict[str, Any]: Analysis information
        """
        if option_name not in self.execution_history:
            return {"success": False, "reason": "No execution data"}
            
        executions = self.execution_history[option_name]
        errors = self.error_history[option_name]
        
        # Compute execution statistics
        success_rate = self._compute_success_rate(executions)
        error_patterns = self._analyze_error_patterns(errors)
        state_coverage = self._compute_state_coverage(executions)
        action_distribution = self._compute_action_distribution(executions)
        
        # Check debug points
        debug_issues = self._check_debug_points(option_name)
        
        return {
            "success_rate": float(success_rate),
            "error_patterns": error_patterns,
            "state_coverage": float(state_coverage),
            "action_distribution": action_distribution,
            "debug_issues": debug_issues
        }
        
    def _compute_success_rate(self, executions: List[Dict[str, Any]]) -> float:
        """Compute success rate of executions."""
        if not executions:
            return 0.0
            
        successes = sum(
            1 for exec in executions
            if exec["result"].get("success", False)
        )
        
        return float(successes) / len(executions)
        
    def _analyze_error_patterns(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze patterns in errors."""
        if not errors:
            return []
            
        # Group errors by type
        error_groups = defaultdict(list)
        for error in errors:
            error_type = error["error"].get("type", "unknown")
            error_groups[error_type].append(error)
            
        # Analyze each error group
        patterns = []
        for error_type, group_errors in error_groups.items():
            pattern = {
                "type": error_type,
                "count": len(group_errors),
                "common_states": self._find_common_states(group_errors),
                "common_actions": self._find_common_actions(group_errors)
            }
            patterns.append(pattern)
            
        return patterns
        
    def _find_common_states(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find common state features in errors."""
        if not errors:
            return {}
            
        # Get all state features
        state_features = set(errors[0]["state"].keys())
        for error in errors[1:]:
            state_features &= set(error["state"].keys())
            
        # Compute statistics for common features
        common_states = {}
        for feature in state_features:
            values = [error["state"][feature] for error in errors]
            common_states[feature] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
            
        return common_states
        
    def _find_common_actions(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find common action features in errors."""
        if not errors:
            return {}
            
        # Get all action features
        action_features = set(errors[0]["action"].keys())
        for error in errors[1:]:
            action_features &= set(error["action"].keys())
            
        # Compute statistics for common features
        common_actions = {}
        for feature in action_features:
            values = [error["action"][feature] for error in errors]
            common_actions[feature] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
            
        return common_actions
        
    def _compute_state_coverage(self, executions: List[Dict[str, Any]]) -> float:
        """Compute coverage of state space."""
        if not executions:
            return 0.0
            
        # Get unique states
        unique_states = set()
        for exec in executions:
            state_key = self._get_state_key(exec["state"])
            unique_states.add(state_key)
            
        # Estimate coverage (this is a simplified measure)
        return float(len(unique_states)) / len(executions)
        
    def _compute_action_distribution(self, executions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute distribution of actions."""
        if not executions:
            return {}
            
        # Count action occurrences
        action_counts = defaultdict(int)
        for exec in executions:
            action_key = self._get_action_key(exec["action"])
            action_counts[action_key] += 1
            
        # Normalize counts
        total = sum(action_counts.values())
        distribution = {
            action: count / total
            for action, count in action_counts.items()
        }
        
        return distribution
        
    def _check_debug_points(self, option_name: str) -> List[Dict[str, Any]]:
        """Check debug points for issues."""
        if option_name not in self.debug_points:
            return []
            
        issues = []
        for point in self.debug_points[option_name]:
            if not self._verify_debug_point(point):
                issues.append({
                    "point": point,
                    "reason": "Debug point verification failed"
                })
                
        return issues
        
    def _verify_debug_point(self, debug_point: Dict[str, Any]) -> bool:
        """Verify a debug point."""
        # Check required fields
        required_fields = ["type", "condition", "action"]
        if not all(field in debug_point for field in required_fields):
            return False
            
        # Check condition validity
        condition = debug_point["condition"]
        if not isinstance(condition, dict):
            return False
            
        # Check action validity
        action = debug_point["action"]
        if not isinstance(action, dict):
            return False
            
        return True
        
    def _get_state_key(self, state: Dict[str, Any]) -> str:
        """Get a unique key for a state."""
        key_parts = []
        for k, v in sorted(state.items()):
            if isinstance(v, (int, float)):
                v = round(v, 2)
            key_parts.append(f"{k}:{v}")
        return "|".join(key_parts)
        
    def _get_action_key(self, action: Dict[str, Any]) -> str:
        """Get a unique key for an action."""
        key_parts = []
        for k, v in sorted(action.items()):
            if isinstance(v, (int, float)):
                v = round(v, 2)
            key_parts.append(f"{k}:{v}")
        return "|".join(key_parts)
        
    def get_debug_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about option debugging.
        
        Returns:
            Dict[str, Any]: Dictionary of debug statistics
        """
        stats = {
            "num_options": len(self.execution_history),
            "total_executions": sum(
                len(history) for history in self.execution_history.values()
            ),
            "total_errors": sum(
                len(errors) for errors in self.error_history.values()
            ),
            "total_debug_points": sum(
                len(points) for points in self.debug_points.values()
            )
        }
        
        return stats
        
    def reset(self):
        """Reset debug state."""
        self.execution_history.clear()
        self.error_history.clear()
        self.debug_points.clear() 