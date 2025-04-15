from typing import Dict, Any, List, Set
import numpy as np

class OptionCoordinator:
    """Coordinates options between multiple agents in the environment."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_states = {}
        self.option_assignments = {}
        self.coordination_history = {}
        
    def coordinate_options(self, agent_id: str, state: Dict[str, Any],
                         available_options: List[str]) -> str:
        """
        Coordinate option selection for an agent.
        
        Args:
            agent_id: ID of the agent
            state: Current state of the agent
            available_options: List of available options
            
        Returns:
            str: Selected option name
        """
        self.agent_states[agent_id] = state
        
        # Get current option assignments
        current_assignments = self._get_current_assignments()
        
        # Check for conflicts
        conflicts = self._detect_conflicts(agent_id, available_options, current_assignments)
        
        if conflicts:
            # Resolve conflicts
            selected_option = self._resolve_conflicts(agent_id, conflicts, current_assignments)
        else:
            # No conflicts, select based on state
            selected_option = self._select_best_option(agent_id, available_options, state)
            
        # Update coordination history
        self._update_coordination_history(agent_id, selected_option)
        
        return selected_option
        
    def _get_current_assignments(self) -> Dict[str, str]:
        """Get current option assignments for all agents."""
        return {aid: self.option_assignments.get(aid, None)
                for aid in self.agent_states.keys()}
                
    def _detect_conflicts(self, agent_id: str, available_options: List[str],
                        current_assignments: Dict[str, str]) -> Set[str]:
        """Detect conflicts between agent options."""
        conflicts = set()
        
        for option in available_options:
            # Check if option is already assigned to another agent
            if option in current_assignments.values():
                conflicts.add(option)
                
            # Check for spatial conflicts
            if self._has_spatial_conflict(agent_id, option, current_assignments):
                conflicts.add(option)
                
        return conflicts
        
    def _has_spatial_conflict(self, agent_id: str, option: str,
                            current_assignments: Dict[str, str]) -> bool:
        """Check for spatial conflicts between agents."""
        agent_state = self.agent_states[agent_id]
        
        for other_id, other_option in current_assignments.items():
            if other_id == agent_id:
                continue
                
            other_state = self.agent_states[other_id]
            
            # Check if agents would be too close
            if self._agents_too_close(agent_state, other_state):
                return True
                
        return False
        
    def _agents_too_close(self, state1: Dict[str, Any],
                         state2: Dict[str, Any]) -> bool:
        """Check if two agents are too close to each other."""
        pos1 = state1["agent_position"]
        pos2 = state2["agent_position"]
        min_distance = self.config.get("min_agent_distance", 5.0)
        
        return np.linalg.norm(pos1 - pos2) < min_distance
        
    def _resolve_conflicts(self, agent_id: str, conflicts: Set[str],
                         current_assignments: Dict[str, str]) -> str:
        """Resolve option conflicts between agents."""
        # Get non-conflicting options
        available_options = set(self.agent_states[agent_id].get("available_options", []))
        non_conflicting = available_options - conflicts
        
        if non_conflicting:
            # Select best non-conflicting option
            return self._select_best_option(agent_id, list(non_conflicting),
                                          self.agent_states[agent_id])
        else:
            # If no non-conflicting options, select least conflicting
            return min(conflicts, key=lambda x: self._get_conflict_score(agent_id, x))
            
    def _get_conflict_score(self, agent_id: str, option: str) -> float:
        """Calculate conflict score for an option."""
        score = 0.0
        
        # Add penalty for option being used by others
        for other_id, other_option in self.option_assignments.items():
            if other_id != agent_id and other_option == option:
                score += 1.0
                
        # Add penalty for spatial conflicts
        agent_state = self.agent_states[agent_id]
        for other_id, other_state in self.agent_states.items():
            if other_id != agent_id:
                if self._agents_too_close(agent_state, other_state):
                    score += 0.5
                    
        return score
        
    def _select_best_option(self, agent_id: str, available_options: List[str],
                          state: Dict[str, Any]) -> str:
        """Select the best option based on state and coordination history."""
        scores = {}
        
        for option in available_options:
            # Base score from state
            score = self._get_option_score(option, state)
            
            # Adjust based on coordination history
            history_score = self._get_history_score(agent_id, option)
            score += history_score
            
            scores[option] = score
            
        return max(scores.items(), key=lambda x: x[1])[0]
        
    def _get_option_score(self, option: str, state: Dict[str, Any]) -> float:
        """Calculate base score for an option based on state."""
        score = 0.0
        
        # Add score based on state features
        if option == "attack" and "flag_distance" in state:
            score += 1.0 / (1.0 + state["flag_distance"])
        elif option == "defend" and "own_flag_distance" in state:
            score += 1.0 / (1.0 + state["own_flag_distance"])
        elif option == "patrol" and "patrol_points" in state:
            score += 0.5
            
        return score
        
    def _get_history_score(self, agent_id: str, option: str) -> float:
        """Calculate score based on coordination history."""
        if agent_id not in self.coordination_history:
            return 0.0
            
        history = self.coordination_history[agent_id]
        if not history:
            return 0.0
            
        # Calculate average success rate for this option
        option_history = [h for h in history if h["option"] == option]
        if not option_history:
            return 0.0
            
        return np.mean([h["success_rate"] for h in option_history])
        
    def _update_coordination_history(self, agent_id: str, option: str):
        """Update coordination history for an agent."""
        if agent_id not in self.coordination_history:
            self.coordination_history[agent_id] = []
            
        self.coordination_history[agent_id].append({
            "option": option,
            "timestamp": len(self.coordination_history[agent_id]),
            "success_rate": 0.0  # This should be updated with actual performance
        })
        
        # Keep only recent history
        max_history = self.config.get("max_coordination_history", 100)
        if len(self.coordination_history[agent_id]) > max_history:
            self.coordination_history[agent_id] = self.coordination_history[agent_id][-max_history:]
            
    def reset(self):
        """Reset coordination state."""
        self.agent_states = {}
        self.option_assignments = {}
        self.coordination_history = {} 