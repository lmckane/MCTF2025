from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np

class BasePolicy(ABC):
    """Base class for reinforcement learning policies."""
    
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
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """
        Update the policy based on the transition.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        pass

class BaseHierarchicalPolicy(ABC):
    """Base class for hierarchical reinforcement learning policies."""
    
    def __init__(self, options: List[str]):
        """
        Initialize the hierarchical policy.
        
        Args:
            options: List of available high-level behaviors/options
        """
        self.options = options
        self.current_option = None
        self.option_history = []
        
    @abstractmethod
    def select_option(self, state: Dict[str, Any]) -> str:
        """
        Select the next high-level option based on the current state.
        
        Args:
            state: Current environment state
            
        Returns:
            str: Selected option name
        """
        pass
    
    @abstractmethod
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get the action to take based on the current state and selected option.
        
        Args:
            state: Current environment state
            
        Returns:
            np.ndarray: Action to take
        """
        pass
    
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """
        Update the policy based on the transition.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        pass 