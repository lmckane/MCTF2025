import numpy as np
from typing import Dict, Any, Optional
from hrl.options.base import BaseOption

class ReturnToBaseOption(BaseOption):
    """Option for returning to base, especially when carrying the flag."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the return to base option.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - max_stuck_steps: Maximum steps without progress before termination (default: 30)
                - evade_distance: Distance at which to evade opponents (default: 15.0)
                - path_planning: Whether to use path planning (default: True)
                - path_update_freq: How often to update path (default: 5)
                - return_speed: Speed multiplier when returning to base (default: 1.2)
        """
        super().__init__("return_to_base", config)
        self.base_position = None
        self.last_position = None
        self.steps_without_progress = 0
        self.max_stuck_steps = self.config.get('max_stuck_steps', 30)
        self.evade_distance = self.config.get('evade_distance', 15.0)
        self.use_path_planning = self.config.get('path_planning', True)
        self.path_update_freq = self.config.get('path_update_freq', 5)
        self.return_speed = self.config.get('return_speed', 1.2)
        self.path_to_base = []
        self.steps_since_path_update = 0
        
    def initiate(self, state: Dict[str, Any]) -> bool:
        """
        Determine if this option should be initiated in the given state.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to initiate this option
        """
        # Check if agent has flag
        if not hasattr(state['agents'][0], 'has_flag'):
            return False
            
        agent = state['agents'][0]
        has_flag = agent.has_flag if hasattr(agent, 'has_flag') else agent.get('has_flag', False)
        
        # Check if agent is in danger (enemies nearby)
        agent_pos = np.array(agent.position if hasattr(agent, 'position') else agent.get('position', [0, 0]))
        agent_team = agent.team if hasattr(agent, 'team') else agent.get('team', 0)
        
        # Find base position
        self.base_position = np.array(state['team_bases'][agent_team])
        
        # Check flag status - initiate if agent has flag or is near base with low health
        if has_flag:
            return True
            
        # Also consider returning to base if health is low
        agent_health = agent.health if hasattr(agent, 'health') else agent.get('health', 100.0)
        dist_to_base = np.linalg.norm(agent_pos - self.base_position)
        
        if agent_health < 30.0 and dist_to_base < 50.0:
            return True
            
        return False
        
    def terminate(self, state: Dict[str, Any]) -> bool:
        """
        Determine if this option should be terminated.
        
        Args:
            state: Current environment state
            
        Returns:
            bool: Whether to terminate this option
        """
        # Terminate if agent no longer has flag
        agent = state['agents'][0]
        has_flag = agent.has_flag if hasattr(agent, 'has_flag') else agent.get('has_flag', False)
        
        if not has_flag:
            # Check if we're at base - terminate if close enough
            agent_pos = np.array(agent.position if hasattr(agent, 'position') else agent.get('position', [0, 0]))
            agent_team = agent.team if hasattr(agent, 'team') else agent.get('team', 0)
            base_pos = np.array(state['team_bases'][agent_team])
            
            dist_to_base = np.linalg.norm(agent_pos - base_pos)
            if dist_to_base < 5.0:  # Close enough to base
                return True
                
        # Check if stuck
        agent_pos = np.array(agent.position if hasattr(agent, 'position') else agent.get('position', [0, 0]))
        if self.last_position is not None:
            progress = np.linalg.norm(agent_pos - self.last_position)
            if progress < 0.5:  # Little movement
                self.steps_without_progress += 1
            else:
                self.steps_without_progress = 0
                
        self.last_position = agent_pos
        
        if self.steps_without_progress > self.max_stuck_steps:
            return True
            
        return False
        
    def get_action(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get the action to return to base while avoiding opponents.
        
        Args:
            state: Current environment state
            
        Returns:
            np.ndarray: Action vector (2D direction)
        """
        agent = state['agents'][0]
        agent_pos = np.array(agent.position if hasattr(agent, 'position') else agent.get('position', [0, 0]))
        agent_team = agent.team if hasattr(agent, 'team') else agent.get('team', 0)
        
        # Update or get base position
        base_pos = np.array(state['team_bases'][agent_team])
        self.base_position = base_pos
        
        # Update path planning
        self.steps_since_path_update += 1
        if self.use_path_planning and (not self.path_to_base or self.steps_since_path_update >= self.path_update_freq):
            self.path_to_base = self._plan_path_to_base(state)
            self.steps_since_path_update = 0
            
        # Get movement direction - either from path or direct
        if self.path_to_base and len(self.path_to_base) > 0 and self.use_path_planning:
            # Follow path
            next_point = np.array(self.path_to_base[0])
            if np.linalg.norm(agent_pos - next_point) < 5.0:
                self.path_to_base.pop(0)
                if len(self.path_to_base) > 0:
                    next_point = np.array(self.path_to_base[0])
                    
            direction = next_point - agent_pos
        else:
            # Direct path to base
            direction = base_pos - agent_pos
            
        # Check for nearby opponents to evade
        nearby_opponents = []
        for other_agent in state['agents'][1:]:  # Skip self
            other_team = other_agent.team if hasattr(other_agent, 'team') else other_agent.get('team', 0)
            if other_team != agent_team:  # Enemy
                other_pos = np.array(other_agent.position if hasattr(other_agent, 'position') else other_agent.get('position', [0, 0]))
                dist = np.linalg.norm(agent_pos - other_pos)
                if dist < self.evade_distance:
                    nearby_opponents.append((other_pos, dist))
                    
        # Adjust direction to avoid opponents
        if nearby_opponents:
            evasion_vector = np.zeros(2)
            for enemy_pos, dist in nearby_opponents:
                # Vector away from enemy, weighted by proximity
                away_vector = agent_pos - enemy_pos
                weight = 1.0 / max(0.1, dist)  # Closer enemies have more influence
                evasion_vector += away_vector * weight
                
            # Blend path to base with evasion
            evasion_weight = min(0.7, 1.0 - (min(dist for _, dist in nearby_opponents) / self.evade_distance))
            blended_direction = (1.0 - evasion_weight) * direction + evasion_weight * evasion_vector
            direction = blended_direction
        
        # Normalize and scale
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction) * self.return_speed
            
        return direction
        
    def _plan_path_to_base(self, state: Dict[str, Any]) -> list:
        """
        Plan a path to base avoiding obstacles and enemies.
        
        Args:
            state: Current environment state
            
        Returns:
            list: List of waypoints to follow
        """
        # Simple implementation - could be enhanced with A* or similar
        agent = state['agents'][0]
        agent_pos = np.array(agent.position if hasattr(agent, 'position') else agent.get('position', [0, 0]))
        agent_team = agent.team if hasattr(agent, 'team') else agent.get('team', 0)
        base_pos = np.array(state['team_bases'][agent_team])
        
        # Direct path first
        path = [base_pos]
        
        # Enemy positions to avoid
        enemy_positions = []
        for other_agent in state['agents'][1:]:
            other_team = other_agent.team if hasattr(other_agent, 'team') else other_agent.get('team', 0)
            if other_team != agent_team:  # Enemy
                other_pos = np.array(other_agent.position if hasattr(other_agent, 'position') else other_agent.get('position', [0, 0]))
                enemy_positions.append(other_pos)
                
        # Check if direct path has enemies nearby
        direct_path_safe = True
        for enemy_pos in enemy_positions:
            # Check if enemy is close to the direct path
            direct_vector = base_pos - agent_pos
            direct_dist = np.linalg.norm(direct_vector)
            if direct_dist > 0:
                direct_unit = direct_vector / direct_dist
                enemy_to_agent = agent_pos - enemy_pos
                
                # Project enemy position onto path line
                projection = np.dot(enemy_to_agent, direct_unit)
                if 0 <= projection <= direct_dist:
                    # Calculate perpendicular distance from enemy to path
                    perp_vector = enemy_to_agent - direct_unit * projection
                    perp_dist = np.linalg.norm(perp_vector)
                    
                    if perp_dist < 10.0:  # Enemy too close to path
                        direct_path_safe = False
                        break
                        
        # If direct path isn't safe, add intermediate waypoints
        if not direct_path_safe:
            # Find midpoint of map
            map_size = np.array(state.get('map_size', [100, 100]))
            map_center = map_size / 2
            
            # Calculate perpendicular vector to direct path
            perp_vector = np.array([-direct_vector[1], direct_vector[0]])
            if np.linalg.norm(perp_vector) > 0:
                perp_vector = perp_vector / np.linalg.norm(perp_vector) * 20.0  # Offset by 20 units
                
                # Try both sides of the direct path
                left_waypoint = (agent_pos + base_pos) / 2 + perp_vector
                right_waypoint = (agent_pos + base_pos) / 2 - perp_vector
                
                # Choose the waypoint closer to map center
                if np.linalg.norm(left_waypoint - map_center) < np.linalg.norm(right_waypoint - map_center):
                    path.insert(0, left_waypoint)
                else:
                    path.insert(0, right_waypoint)
                    
        return path
        
    def reset(self):
        """Reset internal state."""
        super().reset()
        self.base_position = None
        self.last_position = None
        self.steps_without_progress = 0
        self.path_to_base = []
        self.steps_since_path_update = 0 