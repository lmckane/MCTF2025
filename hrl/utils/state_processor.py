from typing import Dict, Any, List, Tuple
import numpy as np
from dataclasses import dataclass
from hrl.utils.team_coordinator import TeamCoordinator, AgentRole

@dataclass
class AgentState:
    """Structured state for a single agent."""
    position: np.ndarray
    velocity: np.ndarray
    heading: float
    has_flag: bool
    is_tagged: bool
    health: float
    team: int
    id: int

@dataclass
class ProcessedState:
    """Processed state information."""
    agent_positions: np.ndarray  # Shape: (num_agents, 2)
    agent_velocities: np.ndarray  # Shape: (num_agents, 2)
    agent_flags: np.ndarray  # Shape: (num_agents,)
    agent_tags: np.ndarray  # Shape: (num_agents,)
    agent_health: np.ndarray  # Shape: (num_agents,)
    agent_teams: np.ndarray  # Shape: (num_agents,)
    agent_territory: np.ndarray  # Shape: (num_agents,)
    flag_positions: np.ndarray  # Shape: (num_flags, 2)
    flag_captured: np.ndarray  # Shape: (num_flags,)
    flag_teams: np.ndarray  # Shape: (num_flags,)
    base_positions: np.ndarray  # Shape: (num_teams, 2)
    step_count: int
    game_state: int
    # Team coordination features
    agent_roles: np.ndarray  # Shape: (num_agents,)
    recommended_target: np.ndarray  # Shape: (2,)
    our_flag_threat: float
    enemy_flag_captured: bool
    our_flag_captured: bool

class StateProcessor:
    """Processes raw environment states into a format suitable for the policy."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the state processor."""
        self.config = config
        self.max_agents = config.get('max_agents', 6)
        self.num_flags = config.get('num_flags', 2)
        self.num_teams = config.get('num_teams', 2)
        self.normalize = config.get('normalize', True)
        self.norm_range = config.get('norm_range', [-1, 1])
        
        # Initialize team coordinator for team-based decision making
        self.team_coordinator = TeamCoordinator({
            'num_agents': config.get('num_agents', 3),
            'role_update_frequency': config.get('role_update_frequency', 10),
            'team_id': 0  # Our agent is always on team 0
        })
        
    def reset(self):
        """Reset the state processor for a new episode."""
        self.team_coordinator.reset()
        
    def process_state(self, state: Dict[str, Any]) -> ProcessedState:
        """Process raw state into structured format with team coordination."""
        # Update team coordinator with the current state
        self.team_coordinator.update_roles(state)
        
        # Extract agent information
        num_agents = len(state['agents'])
        agent_positions = np.zeros((self.max_agents, 2))
        agent_velocities = np.zeros((self.max_agents, 2))
        agent_flags = np.zeros(self.max_agents)
        agent_tags = np.zeros(self.max_agents)
        agent_teams = np.zeros(self.max_agents)
        agent_health = np.zeros(self.max_agents)
        agent_roles = np.zeros(self.max_agents)
        
        # Process agent information with coordination data
        for i, agent in enumerate(state['agents'][:self.max_agents]):
            agent_id = agent.get('id', i)
            agent_positions[i] = self._normalize_position(agent['position'])
            agent_velocities[i] = self._normalize_velocity(agent['velocity'])
            agent_flags[i] = float(agent['has_flag'])
            agent_tags[i] = float(agent['is_tagged'])
            agent_teams[i] = agent['team']
            agent_health[i] = agent['health'] / 100.0
            
            # Add role information for team members
            if agent['team'] == 0:  # Our team
                role = self.team_coordinator.get_agent_role(agent_id)
                agent_roles[i] = role.value  # Store the role enum value
            else:
                agent_roles[i] = -1  # Default for enemy agents
            
        # Extract flag information
        flag_positions = np.zeros((self.num_flags, 2))
        flag_captured = np.zeros(self.num_flags)
        flag_teams = np.zeros(self.num_flags)
        
        for i, flag in enumerate(state['flags']):
            flag_positions[i] = self._normalize_position(flag['position'])
            flag_captured[i] = float(flag['is_captured'])
            flag_teams[i] = flag['team']
            
        # Extract base positions
        base_positions = np.zeros((self.num_teams, 2))
        for team, pos in state['team_bases'].items():
            base_positions[team] = self._normalize_position(pos)
            
        # Get coordination data for the primary agent (agent 0)
        primary_agent_id = state['agents'][0].get('id', 0) if state['agents'] else 0
        coordination_data = self.team_coordinator.get_coordination_data(primary_agent_id, state)
        
        # Get recommended target position based on role
        recommended_target = self._normalize_position(coordination_data['recommended_target'])
        
        # Extract threat levels and flag status
        our_flag_threat = coordination_data['own_flag_threat']
        our_flag_captured = coordination_data['our_flag_captured']
        enemy_flag_captured = coordination_data['enemy_flag_captured']
            
        return ProcessedState(
            agent_positions=agent_positions,
            agent_velocities=agent_velocities,
            agent_flags=agent_flags,
            agent_tags=agent_tags,
            agent_health=agent_health,
            agent_teams=agent_teams,
            agent_territory=np.zeros(self.max_agents),
            flag_positions=flag_positions,
            flag_captured=flag_captured,
            flag_teams=flag_teams,
            base_positions=base_positions,
            step_count=state['step_count'],
            game_state=state['game_state'].value,
            # Team coordination features
            agent_roles=agent_roles,
            recommended_target=recommended_target,
            our_flag_threat=our_flag_threat,
            enemy_flag_captured=enemy_flag_captured,
            our_flag_captured=our_flag_captured
        )
        
    def _normalize_position(self, position: np.ndarray) -> np.ndarray:
        """Normalize position coordinates."""
        if not self.normalize:
            return position
            
        map_size = self.config.get('map_size', [100, 100])
        position = np.array(position, dtype=np.float32)
        position = position / np.array(map_size)
        if self.norm_range != [0, 1]:
            position = (position * (self.norm_range[1] - self.norm_range[0]) + 
                       self.norm_range[0])
        return position
        
    def _normalize_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Normalize velocity values."""
        if not self.normalize:
            return velocity
            
        velocity = np.array(velocity, dtype=np.float32)
        max_velocity = self.config.get('max_velocity', 1.0)
        velocity = velocity / max_velocity
        if self.norm_range != [-1, 1]:
            velocity = (velocity * (self.norm_range[1] - self.norm_range[0]) + 
                       self.norm_range[0])
        return velocity
        
    def get_state_size(self) -> int:
        """Get the size of the processed state vector."""
        # Calculate total size of all features
        size = (
            self.max_agents * 9 +  # Agent features (position[2], velocity[2], flags[1], tags[1], team[1], health[1], role[1])
            self.num_flags * 4 +   # Flag features (position[2], captured[1], team[1])
            self.num_teams * 2 +   # Base positions (position[2])
            2 +                    # Game state (step_count[1], game_state[1])
            5                      # Coordination (recommended_target[2], our_flag_threat[1], our_flag_captured[1], enemy_flag_captured[1])
        )
        return size
        
    def get_action_size(self) -> int:
        """Get the size of the action space."""
        return 2  # 2D movement 