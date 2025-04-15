from typing import Dict, Any, List, Tuple
import numpy as np

class OptionPlanner:
    """Plans sequences of options for agents."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.planning_horizon = config.get("planning_horizon", 5)
        self.discount_factor = config.get("discount_factor", 0.99)
        self.option_transitions = {}
        
    def plan_sequence(self, current_state: Dict[str, Any],
                     available_options: List[str]) -> List[Tuple[str, float]]:
        """
        Plan a sequence of options.
        
        Args:
            current_state: Current environment state
            available_options: List of available options
            
        Returns:
            List[Tuple[str, float]]: List of (option, expected_reward) pairs
        """
        sequences = []
        
        # Generate possible sequences
        for option in available_options:
            sequence = self._generate_sequence(current_state, option, [])
            sequences.append(sequence)
            
        # Evaluate sequences
        evaluated_sequences = []
        for sequence in sequences:
            expected_reward = self._evaluate_sequence(sequence, current_state)
            evaluated_sequences.append((sequence, expected_reward))
            
        # Sort by expected reward
        evaluated_sequences.sort(key=lambda x: x[1], reverse=True)
        
        return evaluated_sequences
        
    def _generate_sequence(self, state: Dict[str, Any], current_option: str,
                         current_sequence: List[str]) -> List[str]:
        """Generate a sequence of options."""
        sequence = current_sequence + [current_option]
        
        if len(sequence) >= self.planning_horizon:
            return sequence
            
        # Get possible next options
        next_options = self._get_next_options(current_option, state)
        
        if not next_options:
            return sequence
            
        # Recursively generate sequences
        best_sequence = sequence
        best_reward = float('-inf')
        
        for next_option in next_options:
            new_sequence = self._generate_sequence(state, next_option, sequence)
            reward = self._evaluate_sequence(new_sequence, state)
            
            if reward > best_reward:
                best_sequence = new_sequence
                best_reward = reward
                
        return best_sequence
        
    def _get_next_options(self, current_option: str,
                         state: Dict[str, Any]) -> List[str]:
        """Get possible next options based on current option and state."""
        if current_option not in self.option_transitions:
            return []
            
        transitions = self.option_transitions[current_option]
        next_options = []
        
        for next_option, conditions in transitions.items():
            if self._check_conditions(conditions, state):
                next_options.append(next_option)
                
        return next_options
        
    def _check_conditions(self, conditions: Dict[str, Any],
                        state: Dict[str, Any]) -> bool:
        """Check if conditions for option transition are met."""
        for key, value in conditions.items():
            if key not in state:
                return False
                
            if isinstance(value, (int, float)):
                if state[key] != value:
                    return False
            elif isinstance(value, tuple):
                min_val, max_val = value
                if not (min_val <= state[key] <= max_val):
                    return False
                    
        return True
        
    def _evaluate_sequence(self, sequence: List[str],
                         initial_state: Dict[str, Any]) -> float:
        """Evaluate a sequence of options."""
        total_reward = 0.0
        current_state = initial_state.copy()
        
        for i, option in enumerate(sequence):
            # Get expected reward for this option
            reward = self._get_expected_reward(option, current_state)
            
            # Discount future rewards
            total_reward += reward * (self.discount_factor ** i)
            
            # Update state (simulated)
            current_state = self._simulate_state_transition(
                current_state, option
            )
            
        return total_reward
        
    def _get_expected_reward(self, option: str,
                           state: Dict[str, Any]) -> float:
        """Get expected reward for an option in a given state."""
        reward = 0.0
        
        # Add reward based on option type
        if option == "attack":
            if "flag_distance" in state:
                reward += 1.0 / (1.0 + state["flag_distance"])
        elif option == "defend":
            if "own_flag_distance" in state:
                reward += 1.0 / (1.0 + state["own_flag_distance"])
        elif option == "patrol":
            reward += 0.5
            
        # Add reward based on state features
        if "has_flag" in state and state["has_flag"]:
            reward += 2.0
            
        if "opponents_nearby" in state and state["opponents_nearby"]:
            reward -= 1.0
            
        return reward
        
    def _simulate_state_transition(self, state: Dict[str, Any],
                                 option: str) -> Dict[str, Any]:
        """Simulate state transition after executing an option."""
        new_state = state.copy()
        
        # Simulate position changes
        if "agent_position" in state:
            if option == "attack":
                new_state["agent_position"] += np.array([1.0, 0.0])
            elif option == "defend":
                new_state["agent_position"] -= np.array([1.0, 0.0])
                
        # Simulate flag status
        if option == "attack" and "flag_distance" in state:
            if state["flag_distance"] < 1.0:
                new_state["has_flag"] = True
                
        return new_state
        
    def update_transitions(self, option: str,
                         transitions: Dict[str, Dict[str, Any]]):
        """
        Update transition rules for an option.
        
        Args:
            option: Name of the option
            transitions: Dictionary mapping next options to their conditions
        """
        self.option_transitions[option] = transitions
        
    def reset(self):
        """Reset planning state."""
        self.option_transitions = {} 