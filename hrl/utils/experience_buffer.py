from typing import Dict, Any, List, Tuple
import numpy as np
from collections import deque

class ExperienceBuffer:
    """Stores and manages experience replay data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_size = config.get("buffer_size", 10000)
        self.buffer = deque(maxlen=self.max_size)
        self.batch_size = config.get("batch_size", 32)
        
    def add(self, experience: Tuple[Dict[str, Any], np.ndarray, float, Dict[str, Any], bool]):
        """
        Add an experience to the buffer.
        
        Args:
            experience: Tuple of (state, action, reward, next_state, done)
        """
        self.buffer.append(experience)
        
    def sample(self, batch_size: int = None) -> List[Tuple]:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            List[Tuple]: Sampled experiences
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.buffer) < batch_size:
            return list(self.buffer)
            
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
        
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        
    def __len__(self) -> int:
        """Get current size of buffer."""
        return len(self.buffer)
        
    def get_all(self) -> List[Tuple]:
        """Get all experiences in buffer."""
        return list(self.buffer)
        
    def get_recent(self, n: int) -> List[Tuple]:
        """
        Get most recent n experiences.
        
        Args:
            n: Number of recent experiences to get
            
        Returns:
            List[Tuple]: Recent experiences
        """
        return list(self.buffer)[-n:]
        
    def get_priority(self, experience: Tuple) -> float:
        """
        Calculate priority for experience.
        
        Args:
            experience: Experience tuple
            
        Returns:
            float: Priority value
        """
        _, _, reward, _, done = experience
        
        # Higher priority for:
        # 1. Terminal states
        # 2. High magnitude rewards
        # 3. Recent experiences
        priority = 1.0
        if done:
            priority *= 2.0
        priority *= (1.0 + abs(reward))
        return priority
        
    def prioritized_sample(self, batch_size: int = None) -> List[Tuple]:
        """
        Sample experiences with priority.
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            List[Tuple]: Sampled experiences
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.buffer) < batch_size:
            return list(self.buffer)
            
        # Calculate priorities
        priorities = [self.get_priority(exp) for exp in self.buffer]
        probs = np.array(priorities) / sum(priorities)
        
        # Sample with priority
        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probs)
        return [self.buffer[i] for i in indices] 