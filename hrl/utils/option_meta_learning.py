from typing import Dict, Any, List, Tuple
import numpy as np
from collections import defaultdict

class OptionMetaLearner:
    """Implements meta-learning capabilities for options."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.meta_learning_rate = config.get("meta_learning_rate", 0.001)
        self.task_memory = defaultdict(list)
        self.option_performance = defaultdict(dict)
        self.task_similarity = {}
        
    def update_task_memory(self, task_id: str, 
                          option_name: str,
                          performance: float,
                          state: Dict[str, Any]):
        """
        Update memory with task performance.
        
        Args:
            task_id: Identifier for the task
            option_name: Name of the option
            performance: Performance metric
            state: State when option was executed
        """
        self.task_memory[task_id].append({
            "option": option_name,
            "performance": performance,
            "state": state
        })
        
    def compute_task_similarity(self, task1: str, task2: str) -> float:
        """
        Compute similarity between two tasks.
        
        Args:
            task1: First task identifier
            task2: Second task identifier
            
        Returns:
            float: Similarity score between 0 and 1
        """
        if task1 not in self.task_memory or task2 not in self.task_memory:
            return 0.0
            
        # Get state distributions
        states1 = [m["state"] for m in self.task_memory[task1]]
        states2 = [m["state"] for m in self.task_memory[task2]]
        
        # Compute similarity based on state distributions
        similarity = self._compute_state_similarity(states1, states2)
        self.task_similarity[(task1, task2)] = similarity
        return similarity
        
    def _compute_state_similarity(self, states1: List[Dict[str, Any]],
                                states2: List[Dict[str, Any]]) -> float:
        """Compute similarity between state distributions."""
        # Convert states to feature vectors
        features1 = self._extract_features(states1)
        features2 = self._extract_features(states2)
        
        # Compute cosine similarity
        if len(features1) == 0 or len(features2) == 0:
            return 0.0
            
        # Normalize features
        features1 = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
        features2 = features2 / np.linalg.norm(features2, axis=1, keepdims=True)
        
        # Compute similarity
        similarity = np.mean(np.dot(features1, features2.T))
        return float(similarity)
        
    def _extract_features(self, states: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features from states."""
        features = []
        for state in states:
            # Extract numerical features
            feature_vector = []
            for k, v in sorted(state.items()):
                if isinstance(v, (int, float)):
                    feature_vector.append(v)
            features.append(feature_vector)
        return np.array(features)
        
    def transfer_option(self, source_task: str, target_task: str,
                       option_name: str) -> Dict[str, Any]:
        """
        Transfer option from source to target task.
        
        Args:
            source_task: Source task identifier
            target_task: Target task identifier
            option_name: Name of option to transfer
            
        Returns:
            Dict[str, Any]: Transfer information
        """
        if source_task not in self.task_memory or target_task not in self.task_memory:
            return {"success": False, "reason": "Task not found"}
            
        # Get option performance in source task
        source_performance = self._get_option_performance(source_task, option_name)
        if source_performance is None:
            return {"success": False, "reason": "Option not found in source task"}
            
        # Compute task similarity
        similarity = self.compute_task_similarity(source_task, target_task)
        
        # Estimate transfer performance
        transfer_performance = source_performance * similarity
        
        return {
            "success": True,
            "source_performance": source_performance,
            "similarity": similarity,
            "estimated_performance": transfer_performance
        }
        
    def _get_option_performance(self, task_id: str,
                              option_name: str) -> float:
        """Get average performance of option in task."""
        performances = [
            m["performance"] for m in self.task_memory[task_id]
            if m["option"] == option_name
        ]
        if not performances:
            return None
        return float(np.mean(performances))
        
    def get_meta_learning_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about meta-learning.
        
        Returns:
            Dict[str, Any]: Dictionary of meta-learning statistics
        """
        stats = {
            "num_tasks": len(self.task_memory),
            "total_experiences": sum(len(mem) for mem in self.task_memory.values()),
            "average_task_similarity": 0.0,
            "option_transfer_success_rate": 0.0
        }
        
        # Compute average task similarity
        if len(self.task_similarity) > 0:
            stats["average_task_similarity"] = float(
                np.mean(list(self.task_similarity.values()))
            )
            
        return stats
        
    def reset(self):
        """Reset meta-learning state."""
        self.task_memory.clear()
        self.option_performance.clear()
        self.task_similarity.clear() 