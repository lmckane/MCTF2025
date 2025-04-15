from typing import Dict, Any, List, Tuple
import numpy as np
from collections import deque

class OptionMemory:
    """Manages memory for options and their execution history."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_size = config.get("memory_size", 1000)
        self.option_memories = {}
        self.episode_buffer = deque(maxlen=self.memory_size)
        
    def store_experience(self, option_name: str, state: Dict[str, Any],
                        action: np.ndarray, reward: float,
                        next_state: Dict[str, Any], done: bool):
        """
        Store an experience in memory.
        
        Args:
            option_name: Name of the option
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        experience = {
            "option": option_name,
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        }
        
        # Store in option-specific memory
        if option_name not in self.option_memories:
            self.option_memories[option_name] = deque(maxlen=self.memory_size)
            
        self.option_memories[option_name].append(experience)
        
        # Store in episode buffer
        self.episode_buffer.append(experience)
        
    def sample_experiences(self, option_name: str = None,
                         batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Sample experiences from memory.
        
        Args:
            option_name: Optional name of option to sample from
            batch_size: Number of experiences to sample
            
        Returns:
            List[Dict[str, Any]]: List of sampled experiences
        """
        if option_name is not None:
            memory = self.option_memories.get(option_name, [])
        else:
            memory = list(self.episode_buffer)
            
        if len(memory) < batch_size:
            return memory
            
        indices = np.random.choice(len(memory), batch_size, replace=False)
        return [memory[i] for i in indices]
        
    def get_option_statistics(self, option_name: str) -> Dict[str, float]:
        """
        Get statistics for an option.
        
        Args:
            option_name: Name of the option
            
        Returns:
            Dict[str, float]: Dictionary of statistics
        """
        if option_name not in self.option_memories:
            return {}
            
        experiences = list(self.option_memories[option_name])
        
        if not experiences:
            return {}
            
        stats = {
            "total_experiences": len(experiences),
            "average_reward": np.mean([e["reward"] for e in experiences]),
            "success_rate": np.mean([1.0 if e["reward"] > 0 else 0.0 
                                   for e in experiences]),
            "average_duration": np.mean([1.0 for e in experiences])
        }
        
        return stats
        
    def get_episode_statistics(self) -> Dict[str, float]:
        """
        Get statistics for the current episode.
        
        Returns:
            Dict[str, float]: Dictionary of statistics
        """
        if not self.episode_buffer:
            return {}
            
        experiences = list(self.episode_buffer)
        
        stats = {
            "total_steps": len(experiences),
            "total_reward": sum(e["reward"] for e in experiences),
            "average_reward": np.mean([e["reward"] for e in experiences]),
            "options_used": len(set(e["option"] for e in experiences))
        }
        
        return stats
        
    def get_option_transitions(self) -> Dict[str, Dict[str, int]]:
        """
        Get transition statistics between options.
        
        Returns:
            Dict[str, Dict[str, int]]: Dictionary of transition counts
        """
        transitions = {}
        
        for i in range(len(self.episode_buffer) - 1):
            current = self.episode_buffer[i]["option"]
            next_opt = self.episode_buffer[i + 1]["option"]
            
            if current not in transitions:
                transitions[current] = {}
                
            if next_opt not in transitions[current]:
                transitions[current][next_opt] = 0
                
            transitions[current][next_opt] += 1
            
        return transitions
        
    def get_state_distribution(self, option_name: str) -> Dict[str, Any]:
        """
        Get state distribution for an option.
        
        Args:
            option_name: Name of the option
            
        Returns:
            Dict[str, Any]: Dictionary of state statistics
        """
        if option_name not in self.option_memories:
            return {}
            
        experiences = list(self.option_memories[option_name])
        
        if not experiences:
            return {}
            
        # Calculate statistics for each state feature
        state_stats = {}
        
        for key in experiences[0]["state"].keys():
            values = [e["state"][key] for e in experiences]
            
            if isinstance(values[0], (int, float)):
                state_stats[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
                
        return state_stats
        
    def clear_episode_buffer(self):
        """Clear the episode buffer."""
        self.episode_buffer.clear()
        
    def reset(self):
        """Reset all memory."""
        self.option_memories = {}
        self.episode_buffer.clear() 