from typing import Dict, Any
import numpy as np
from hrl.options.base import BaseOption

class PatrolOption(BaseOption):
    """Option for patrolling the environment to find opponents or opportunities."""
    
    def __init__(self):
        super().__init__("patrol")
        self.patrol_points = []
        self.current_patrol_index = 0
        self.patrol_radius = 10.0
        self.last_opponent_seen = None
        
    def initiate(self, state: Dict[str, Any]) -> bool:
        """Initiate if no specific task is needed and we're not in danger."""
        agent_has_flag = state.get("agent_has_flag", False)
        agent_is_tagged = state.get("agent_is_tagged", False)
        opponent_positions = state.get("opponent_positions", [])
        
        if not agent_has_flag and not agent_is_tagged and not opponent_positions:
            self._generate_patrol_points(state)
            return True
        return False
        
    def terminate(self, state: Dict[str, Any]) -> bool:
        """Terminate if we see an opponent or if we're tagged."""
        agent_is_tagged = state.get("agent_is_tagged", False)
        opponent_positions = state.get("opponent_positions", [])
        
        if agent_is_tagged or opponent_positions:
            return True
        return False
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """Get action to move to next patrol point."""
        agent_position = state.get("agent_position", None)
        if agent_position is None or not self.patrol_points:
            return np.zeros(2)
            
        # Move to next patrol point
        target = self.patrol_points[self.current_patrol_index]
        direction = target - agent_position
        distance = np.linalg.norm(direction)
        
        if distance < 1.0:  # Reached current patrol point
            self.current_patrol_index = (self.current_patrol_index + 1) % len(self.patrol_points)
            target = self.patrol_points[self.current_patrol_index]
            direction = target - agent_position
            distance = np.linalg.norm(direction)
            
        if distance > 0:
            direction = direction / distance
            
        return direction
        
    def get_reward(self, state: Dict[str, Any], action: np.ndarray, next_state: Dict[str, Any]) -> float:
        """Get reward based on exploration and opponent detection."""
        reward = 0.0
        
        # Reward for moving to new areas
        agent_position = state.get("agent_position", None)
        if agent_position is not None:
            # Check if we're moving towards unexplored areas
            if self._is_moving_to_unexplored(agent_position, action):
                reward += 0.5
                
        # Reward for finding opponents
        opponent_positions = next_state.get("opponent_positions", [])
        if opponent_positions and not state.get("opponent_positions", []):
            reward += 10.0
            
        return reward
        
    def _generate_patrol_points(self, state: Dict[str, Any]):
        """Generate patrol points in unexplored areas."""
        env_bounds = state.get("env_bounds", [100, 100])
        num_points = 8
        
        # Generate points in a grid pattern
        x_points = np.linspace(0, env_bounds[0], int(np.sqrt(num_points)))
        y_points = np.linspace(0, env_bounds[1], int(np.sqrt(num_points)))
        
        self.patrol_points = []
        for x in x_points:
            for y in y_points:
                self.patrol_points.append(np.array([x, y]))
                
    def _is_moving_to_unexplored(self, position: np.ndarray, action: np.ndarray) -> bool:
        """Check if the agent is moving towards unexplored areas."""
        if not self.patrol_points:
            return True
            
        # Calculate distance to nearest patrol point
        distances = [np.linalg.norm(position - point) for point in self.patrol_points]
        min_dist = min(distances)
        
        # If we're far from all patrol points, we're exploring
        return min_dist > self.patrol_radius
        
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, next_state: Dict[str, Any], done: bool):
        """Update patrol strategy based on new information."""
        opponent_positions = state.get("opponent_positions", [])
        if opponent_positions:
            self.last_opponent_seen = opponent_positions[0]
            
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.patrol_points = []
        self.current_patrol_index = 0
        self.patrol_radius = 10.0
        self.last_opponent_seen = None 