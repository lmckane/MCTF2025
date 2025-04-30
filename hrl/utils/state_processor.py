from typing import Dict, Any, List, Tuple, Optional
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
    """Processed state with normalized features."""
    agent_positions: np.ndarray  # Position of each agent [x, y]
    agent_velocities: np.ndarray  # Velocity of each agent [vx, vy]
    agent_flags: np.ndarray  # Whether each agent has the flag (1) or not (0)
    agent_tags: np.ndarray  # Whether each agent is tagged (1) or not (0)
    agent_health: np.ndarray  # Health/energy of each agent [0-1]
    agent_teams: np.ndarray  # Team of each agent (0 or 1)
    agent_ids: np.ndarray  # ID of each agent
    agent_roles: np.ndarray  # Optional role assignment for agents (0=attacker, 1=defender, 2=interceptor)
    flag_positions: np.ndarray  # Position of each flag [x, y]
    flag_captures: np.ndarray  # Whether each flag is captured (1) or not (0)
    flag_teams: np.ndarray  # Team associated with each flag (0 or 1)
    team_scores: np.ndarray  # Scores of each team
    step: int  # Current step count
    map_size: np.ndarray  # Map size [width, height]
    normalized: bool = True  # Whether features are normalized

class StateProcessor:
    """Process raw environment states into a format suitable for learning."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the state processor.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.max_agents = config.get('max_agents', 6)
        self.num_flags = config.get('num_flags', 2)
        self.num_teams = config.get('num_teams', 2)
        self.normalize = config.get('normalize_state', True)
        self.norm_range = config.get('norm_range', [-1, 1])
        self.debug_level = config.get('debug_level', 1)
        
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
        """
        Process raw state into a normalized, feature-rich representation.
        
        Args:
            state: Raw state from environment
            
        Returns:
            processed_state: Processed state with normalized features
        """
        # Update team coordinator with the current state
        self.team_coordinator.update_roles(state)
        
        # Extract basic features from raw state
        map_size = np.array(state.get('map_size', [100, 100]))
        team_scores = np.array(state.get('team_scores', [0, 0]))
        step = state.get('step', 0)
        
        # Process agent information
        agents = state.get('agents', [])
        num_agents = len(agents)
        
        # Initialize arrays for agent features
        agent_positions = np.zeros((num_agents, 2))
        agent_velocities = np.zeros((num_agents, 2))
        agent_flags = np.zeros(num_agents)
        agent_tags = np.zeros(num_agents)
        agent_health = np.ones(num_agents)
        agent_teams = np.zeros(num_agents)
        agent_ids = np.zeros(num_agents)
        agent_roles = np.zeros(num_agents)  # Default roles
        
        # Populate agent features
        for i, agent in enumerate(agents):
            agent_positions[i] = self._normalize_position(agent['position'])
            agent_velocities[i] = self._normalize_velocity(agent['velocity'])
            agent_flags[i] = float(agent['has_flag'])
            agent_tags[i] = float(agent['is_tagged'])
            agent_health[i] = agent['health'] / 100.0
            agent_teams[i] = agent['team']
            agent_ids[i] = agent['id'] if 'id' in agent else i
        
        # Process flag information
        flags = state.get('flags', [])
        num_flags = len(flags)
        
        # Initialize arrays for flag features
        flag_positions = np.zeros((num_flags, 2))
        flag_captures = np.zeros(num_flags)
        flag_teams = np.zeros(num_flags)
        
        # Populate flag features
        for i, flag in enumerate(flags):
            flag_positions[i] = self._normalize_position(flag['position'])
            flag_captures[i] = float(flag['is_captured'])
            flag_teams[i] = flag['team']
        
        # Normalize features if required
        if self.normalize:
            # Normalize positions and velocities
            agent_positions = agent_positions / map_size
            flag_positions = flag_positions / map_size
            agent_velocities = agent_velocities / 5.0  # Assuming max velocity is 5
        
        # Assign strategic roles for team 0 agents (our agents)
        team_0_agents = [i for i, team in enumerate(agent_teams) if team == 0]
        
        if team_0_agents:
            # For simplicity, assign fixed roles by position
            # 0: Attacker - goes for flag
            # 1: Defender - stays near home flag
            # 2: Interceptor - tries to tag enemies with flags
            
            # Find our flag position (team 0)
            our_flag_pos = flag_positions[0] if num_flags > 0 else np.array([0.1, 0.5])
            
            # Assign each agent a role based on their index
            for i, agent_idx in enumerate(team_0_agents):
                if i % 3 == 0:
                    agent_roles[agent_idx] = 0  # Attacker
                elif i % 3 == 1:
                    agent_roles[agent_idx] = 1  # Defender
                else:
                    agent_roles[agent_idx] = 2  # Interceptor
        
        # Create processed state
        processed_state = ProcessedState(
            agent_positions=agent_positions,
            agent_velocities=agent_velocities,
            agent_flags=agent_flags,
            agent_tags=agent_tags,
            agent_health=agent_health,
            agent_teams=agent_teams,
            agent_ids=agent_ids,
            agent_roles=agent_roles,
            flag_positions=flag_positions,
            flag_captures=flag_captures,
            flag_teams=flag_teams,
            team_scores=team_scores,
            step=step,
            map_size=map_size,
            normalized=self.normalize
        )
        
        return processed_state
        
    def _normalize_position(self, position: np.ndarray) -> np.ndarray:
        """Normalize position coordinates."""
        if not self.normalize:
            return position
            
        position = np.array(position, dtype=np.float32)
        position = position / np.array(self.config.get('map_size', [100, 100]))
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