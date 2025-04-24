import numpy as np
import torch
from typing import Dict, Any
from dataclasses import dataclass
from collections import deque
import random

@dataclass
class Experience:
    """Single step experience tuple."""
    state: Dict[str, Any]  # Raw state
    processed_state: torch.Tensor  # Processed state tensor
    action: np.ndarray
    reward: float
    next_state: Dict[str, Any]  # Raw next state
    processed_next_state: torch.Tensor  # Processed next state tensor
    done: bool
    option: int

class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.adversarial_buffer = deque(maxlen=capacity // 2)
        
    def store(self, state: Dict[str, Any], processed_state: torch.Tensor,
             action: np.ndarray, reward: float, 
             next_state: Dict[str, Any], processed_next_state: torch.Tensor,
             done: bool, option: int, is_adversarial: bool = False):
        """Store experience in buffer."""
        experience = Experience(
            state=state,
            processed_state=processed_state,
            action=action,
            reward=reward,
            next_state=next_state,
            processed_next_state=processed_next_state,
            done=done,
            option=option
        )
        if is_adversarial:
            self.adversarial_buffer.append(experience)
        else:
            self.buffer.append(experience)
            
    def sample(self, batch_size: int) -> list[Experience]:
        """Sample a batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
    def sample_adversarial(self, batch_size: int) -> list[Experience]:
        """Sample a batch of adversarial experiences."""
        return random.sample(self.adversarial_buffer, 
                           min(batch_size, len(self.adversarial_buffer)))
        
    def __len__(self) -> int:
        return len(self.buffer) 