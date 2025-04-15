from typing import Dict, Any, List, Tuple
import numpy as np
from collections import defaultdict

class OptionSafety:
    """Ensures safe option execution."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.safety_threshold = config.get("safety_threshold", 0.9)
        self.risk_history = defaultdict(list)
        self.safety_violations = defaultdict(list)
        self.safety_rules = defaultdict(list)
        
    def add_safety_rule(self, option_name: str,
                       rule: Dict[str, Any]):
        """
        Add a safety rule for an option.
        
        Args:
            option_name: Name of the option
            rule: Dictionary containing rule information
        """
        self.safety_rules[option_name].append(rule)
        
    def check_safety(self, option_name: str,
                    state: Dict[str, Any],
                    action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check safety of an option execution.
        
        Args:
            option_name: Name of the option
            state: Current state
            action: Proposed action
            
        Returns:
            Dict[str, Any]: Safety information
        """
        if option_name not in self.safety_rules:
            return {"safe": True, "reason": "No safety rules defined"}
            
        # Check all safety rules
        violations = []
        for rule in self.safety_rules[option_name]:
            violation = self._check_rule(rule, state, action)
            if violation:
                violations.append(violation)
                
        # Compute risk score
        risk_score = self._compute_risk_score(violations)
        
        # Update risk history
        self.risk_history[option_name].append(risk_score)
        
        # Check if safe
        is_safe = risk_score <= self.safety_threshold
        
        if not is_safe:
            self.safety_violations[option_name].append({
                "state": state,
                "action": action,
                "risk_score": risk_score,
                "violations": violations
            })
            
        return {
            "safe": is_safe,
            "risk_score": risk_score,
            "violations": violations
        }
        
    def _check_rule(self, rule: Dict[str, Any],
                   state: Dict[str, Any],
                   action: Dict[str, Any]) -> Dict[str, Any]:
        """Check a single safety rule."""
        rule_type = rule.get("type")
        
        if rule_type == "state_constraint":
            return self._check_state_constraint(rule, state)
        elif rule_type == "action_constraint":
            return self._check_action_constraint(rule, action)
        elif rule_type == "transition_constraint":
            return self._check_transition_constraint(rule, state, action)
        else:
            return None
            
    def _check_state_constraint(self, rule: Dict[str, Any],
                              state: Dict[str, Any]) -> Dict[str, Any]:
        """Check state-based safety constraint."""
        variable = rule.get("variable")
        operator = rule.get("operator")
        value = rule.get("value")
        
        if variable not in state:
            return None
            
        state_value = state[variable]
        violated = False
        
        if operator == "<":
            violated = state_value < value
        elif operator == "<=":
            violated = state_value <= value
        elif operator == ">":
            violated = state_value > value
        elif operator == ">=":
            violated = state_value >= value
        elif operator == "==":
            violated = state_value == value
        elif operator == "!=":
            violated = state_value != value
            
        if violated:
            return {
                "type": "state_constraint",
                "variable": variable,
                "operator": operator,
                "value": value,
                "actual_value": state_value
            }
            
        return None
        
    def _check_action_constraint(self, rule: Dict[str, Any],
                               action: Dict[str, Any]) -> Dict[str, Any]:
        """Check action-based safety constraint."""
        variable = rule.get("variable")
        operator = rule.get("operator")
        value = rule.get("value")
        
        if variable not in action:
            return None
            
        action_value = action[variable]
        violated = False
        
        if operator == "<":
            violated = action_value < value
        elif operator == "<=":
            violated = action_value <= value
        elif operator == ">":
            violated = action_value > value
        elif operator == ">=":
            violated = action_value >= value
        elif operator == "==":
            violated = action_value == value
        elif operator == "!=":
            violated = action_value != value
            
        if violated:
            return {
                "type": "action_constraint",
                "variable": variable,
                "operator": operator,
                "value": value,
                "actual_value": action_value
            }
            
        return None
        
    def _check_transition_constraint(self, rule: Dict[str, Any],
                                   state: Dict[str, Any],
                                   action: Dict[str, Any]) -> Dict[str, Any]:
        """Check transition-based safety constraint."""
        # This would check constraints on state-action pairs
        # For example, ensuring certain actions are only taken in specific states
        return None
        
    def _compute_risk_score(self, violations: List[Dict[str, Any]]) -> float:
        """Compute risk score based on violations."""
        if not violations:
            return 0.0
            
        # Compute risk based on violation severity
        risk_scores = []
        for violation in violations:
            severity = violation.get("severity", 1.0)
            risk_scores.append(severity)
            
        return float(np.mean(risk_scores))
        
    def get_safety_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about option safety.
        
        Returns:
            Dict[str, Any]: Dictionary of safety statistics
        """
        stats = {
            "num_options": len(self.safety_rules),
            "safe_options": 0,
            "average_risk": 0.0,
            "total_violations": sum(
                len(violations) for violations in self.safety_violations.values()
            )
        }
        
        # Compute statistics
        risk_scores = []
        for option in self.risk_history:
            avg_risk = np.mean(self.risk_history[option])
            risk_scores.append(avg_risk)
            if avg_risk <= self.safety_threshold:
                stats["safe_options"] += 1
                
        if risk_scores:
            stats["average_risk"] = float(np.mean(risk_scores))
            
        return stats
        
    def suggest_safety_improvements(self, option_name: str) -> List[Dict[str, Any]]:
        """
        Suggest safety improvements for an option.
        
        Args:
            option_name: Name of the option
            
        Returns:
            List[Dict[str, Any]]: List of improvement suggestions
        """
        if option_name not in self.safety_violations:
            return []
            
        suggestions = []
        violations = self.safety_violations[option_name]
        
        # Group violations by type
        violation_types = defaultdict(list)
        for violation in violations:
            for v in violation["violations"]:
                violation_types[v["type"]].append(v)
                
        # Create suggestions based on violation types
        for v_type, v_list in violation_types.items():
            if v_type == "state_constraint":
                suggestions.append({
                    "type": "state_constraint",
                    "description": f"Address {len(v_list)} state constraint violations",
                    "examples": v_list[:3]  # Show first 3 examples
                })
            elif v_type == "action_constraint":
                suggestions.append({
                    "type": "action_constraint",
                    "description": f"Address {len(v_list)} action constraint violations",
                    "examples": v_list[:3]
                })
                
        return suggestions
        
    def reset(self):
        """Reset safety state."""
        self.risk_history.clear()
        self.safety_violations.clear()
        self.safety_rules.clear() 