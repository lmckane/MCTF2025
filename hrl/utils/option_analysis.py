from typing import Dict, Any, List, Tuple
import numpy as np
from collections import defaultdict
from datetime import datetime

class OptionAnalyzer:
    """Provides analysis capabilities for option execution and performance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.analysis_threshold = config.get("analysis_threshold", 0.5)
        
    def analyze_execution(self, option_name: str,
                        execution_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze option execution.
        
        Args:
            option_name: Name of the option
            execution_logs: List of execution logs
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        if not execution_logs:
            return {"success": False, "reason": "No execution data"}
            
        # Extract states and actions
        states = [log["state"] for log in execution_logs]
        actions = [log["action"] for log in execution_logs]
        results = [log["result"] for log in execution_logs]
        
        # Compute basic statistics
        success_rate = self._compute_success_rate(results)
        state_coverage = self._compute_state_coverage(states)
        action_diversity = self._compute_action_diversity(actions)
        
        # Analyze state transitions
        transition_patterns = self._analyze_transitions(states, actions)
        
        # Analyze performance trends
        performance_trends = self._analyze_performance_trends(results)
        
        return {
            "success_rate": float(success_rate),
            "state_coverage": float(state_coverage),
            "action_diversity": float(action_diversity),
            "transition_patterns": transition_patterns,
            "performance_trends": performance_trends
        }
        
    def analyze_performance(self, option_name: str,
                          performance_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze option performance.
        
        Args:
            option_name: Name of the option
            performance_logs: List of performance logs
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        if not performance_logs:
            return {"success": False, "reason": "No performance data"}
            
        # Extract metrics
        metrics = [log["metrics"] for log in performance_logs]
        timestamps = [
            datetime.fromisoformat(log["timestamp"])
            for log in performance_logs
        ]
        
        # Compute metric statistics
        metric_stats = {}
        for key in metrics[0].keys():
            values = [metric[key] for metric in metrics]
            metric_stats[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values))
            }
            
        # Analyze trends
        trends = self._analyze_metric_trends(metrics, timestamps)
        
        # Analyze correlations
        correlations = self._analyze_metric_correlations(metrics)
        
        return {
            "metric_statistics": metric_stats,
            "trends": trends,
            "correlations": correlations
        }
        
    def analyze_errors(self, option_name: str,
                     error_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze option errors.
        
        Args:
            option_name: Name of the option
            error_logs: List of error logs
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        if not error_logs:
            return {"success": False, "reason": "No error data"}
            
        # Extract errors
        errors = [log["error"] for log in error_logs]
        timestamps = [
            datetime.fromisoformat(log["timestamp"])
            for log in error_logs
        ]
        
        # Group errors by type
        error_groups = defaultdict(list)
        for error in errors:
            error_type = error.get("type", "unknown")
            error_groups[error_type].append(error)
            
        # Analyze error patterns
        error_patterns = {}
        for error_type, group_errors in error_groups.items():
            pattern = {
                "count": len(group_errors),
                "frequency": len(group_errors) / len(errors),
                "common_causes": self._find_common_causes(group_errors),
                "time_distribution": self._analyze_time_distribution(
                    group_errors, timestamps
                )
            }
            error_patterns[error_type] = pattern
            
        return {
            "total_errors": len(errors),
            "error_patterns": error_patterns
        }
        
    def _compute_success_rate(self, results: List[Dict[str, Any]]) -> float:
        """Compute success rate from results."""
        if not results:
            return 0.0
            
        successes = sum(
            1 for result in results
            if result.get("success", False)
        )
        
        return float(successes) / len(results)
        
    def _compute_state_coverage(self, states: List[Dict[str, Any]]) -> float:
        """Compute coverage of state space."""
        if not states:
            return 0.0
            
        # Get unique states
        unique_states = set()
        for state in states:
            state_key = self._get_state_key(state)
            unique_states.add(state_key)
            
        # Estimate coverage (this is a simplified measure)
        return float(len(unique_states)) / len(states)
        
    def _compute_action_diversity(self, actions: List[Dict[str, Any]]) -> float:
        """Compute diversity of actions."""
        if not actions:
            return 0.0
            
        # Get unique actions
        unique_actions = set()
        for action in actions:
            action_key = self._get_action_key(action)
            unique_actions.add(action_key)
            
        # Compute diversity score
        return float(len(unique_actions)) / len(actions)
        
    def _analyze_transitions(self, states: List[Dict[str, Any]],
                           actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze state-action transitions."""
        if len(states) < 2:
            return []
            
        transitions = []
        for i in range(len(states) - 1):
            transition = {
                "from_state": states[i],
                "action": actions[i],
                "to_state": states[i + 1]
            }
            transitions.append(transition)
            
        # Group similar transitions
        transition_groups = defaultdict(list)
        for transition in transitions:
            group_key = self._get_transition_key(transition)
            transition_groups[group_key].append(transition)
            
        # Analyze transition patterns
        patterns = []
        for group_transitions in transition_groups.values():
            if len(group_transitions) > 1:
                pattern = {
                    "count": len(group_transitions),
                    "example": group_transitions[0],
                    "frequency": len(group_transitions) / len(transitions)
                }
                patterns.append(pattern)
                
        return patterns
        
    def _analyze_performance_trends(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if not results:
            return {}
            
        # Extract performance metrics
        metrics = defaultdict(list)
        for result in results:
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    metrics[key].append(value)
                    
        # Analyze trends
        trends = {}
        for key, values in metrics.items():
            if len(values) > 1:
                trend = np.polyfit(range(len(values)), values, 1)[0]
                trends[key] = {
                    "slope": float(trend),
                    "direction": "increasing" if trend > 0 else "decreasing"
                }
                
        return trends
        
    def _analyze_metric_trends(self, metrics: List[Dict[str, float]],
                             timestamps: List[datetime]) -> Dict[str, Any]:
        """Analyze trends in performance metrics."""
        if not metrics or not timestamps:
            return {}
            
        # Convert timestamps to numerical values
        time_values = [
            (ts - timestamps[0]).total_seconds()
            for ts in timestamps
        ]
        
        # Analyze trends for each metric
        trends = {}
        for key in metrics[0].keys():
            values = [metric[key] for metric in metrics]
            if len(values) > 1:
                trend = np.polyfit(time_values, values, 1)[0]
                trends[key] = {
                    "slope": float(trend),
                    "direction": "increasing" if trend > 0 else "decreasing"
                }
                
        return trends
        
    def _analyze_metric_correlations(self, metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Analyze correlations between performance metrics."""
        if not metrics:
            return {}
            
        # Create correlation matrix
        metric_names = list(metrics[0].keys())
        n_metrics = len(metric_names)
        correlations = np.zeros((n_metrics, n_metrics))
        
        for i in range(n_metrics):
            for j in range(n_metrics):
                values_i = [metric[metric_names[i]] for metric in metrics]
                values_j = [metric[metric_names[j]] for metric in metrics]
                correlations[i, j] = np.corrcoef(values_i, values_j)[0, 1]
                
        # Convert to dictionary
        correlation_dict = {}
        for i in range(n_metrics):
            for j in range(i + 1, n_metrics):
                key = f"{metric_names[i]}-{metric_names[j]}"
                correlation_dict[key] = float(correlations[i, j])
                
        return correlation_dict
        
    def _find_common_causes(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find common causes in errors."""
        if not errors:
            return []
            
        # Group errors by cause
        cause_groups = defaultdict(list)
        for error in errors:
            cause = error.get("cause", "unknown")
            cause_groups[cause].append(error)
            
        # Analyze common causes
        common_causes = []
        for cause, group_errors in cause_groups.items():
            if len(group_errors) > 1:
                common_cause = {
                    "cause": cause,
                    "count": len(group_errors),
                    "frequency": len(group_errors) / len(errors),
                    "example": group_errors[0]
                }
                common_causes.append(common_cause)
                
        return sorted(common_causes, key=lambda x: x["count"], reverse=True)
        
    def _analyze_time_distribution(self, errors: List[Dict[str, Any]],
                                 timestamps: List[datetime]) -> Dict[str, Any]:
        """Analyze time distribution of errors."""
        if not errors or not timestamps:
            return {}
            
        # Compute time intervals between errors
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
            
        if not intervals:
            return {}
            
        return {
            "mean_interval": float(np.mean(intervals)),
            "std_interval": float(np.std(intervals)),
            "min_interval": float(np.min(intervals)),
            "max_interval": float(np.max(intervals))
        }
        
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
        
    def _get_transition_key(self, transition: Dict[str, Any]) -> str:
        """Get a unique key for a transition."""
        from_key = self._get_state_key(transition["from_state"])
        action_key = self._get_action_key(transition["action"])
        to_key = self._get_state_key(transition["to_state"])
        return f"{from_key}|{action_key}|{to_key}" 