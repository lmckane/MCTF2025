from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np

class BaseOption(ABC):
    """Base class for all options in the hierarchical reinforcement learning system."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the option.
        
        Args:
            name: Name of the option
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.is_active = False
        self.termination_conditions = []
        
    @abstractmethod
    def initiate(self, state: Dict[str, Any]) -> bool:
        """
        Determine if this option should be initiated in the given state.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to initiate this option
        """
        pass
        
    @abstractmethod
    def terminate(self, state: Dict[str, Any]) -> bool:
        """
        Determine if this option should be terminated in the given state.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to terminate this option
        """
        pass
        
    @abstractmethod
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get the action to take based on the current state.
        
        Args:
            state: Current environment state
            
        Returns:
            np.ndarray: Action to take
        """
        pass
        
    @abstractmethod
    def get_reward(self, state: Dict[str, Any], action: np.ndarray, next_state: Dict[str, Any]) -> float:
        """
        Get the reward for taking the given action in the given state.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            
        Returns:
            float: Reward value
        """
        pass
        
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """
        Update the option based on the transition.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        pass
        
    def reset(self):
        """Reset the option's internal state."""
        self.is_active = False
        self.termination_conditions = [] 