import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum, auto
import random
import math

# Import required game environment classes if they exist
try:
    from hrl.environment.game_env import Agent, Flag, GameState
except ImportError:
    # Define placeholder classes for type hints if imports fail
    class Agent:
        position: np.ndarray
        velocity: np.ndarray
        has_flag: bool
        is_tagged: bool
        team: int
        id: int
        
    class Flag:
        position: np.ndarray
        is_captured: bool
        team: int
        
    class GameState(Enum):
        PLAYING = 0
        WON = 1
        LOST = 2
        DRAW = 3

class AgentRole(Enum):
    """Possible roles for coordinated agents."""
    ATTACKER = auto()  # Focus on capturing flags
    DEFENDER = auto()  # Focus on defending own flag
    INTERCEPTOR = auto()  # Focus on tagging enemies
    SUPPORT = auto()  # Help other agents

class OpponentStrategy:
    """Base class for opponent strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize strategy with config."""
        self.config = config or {}
        self.name = "Base"
        self.difficulty = self.config.get('difficulty', 0.5)
        
    def get_action(self, agent: Agent, state: Dict[str, Any]) -> np.ndarray:
        """Get action for agent based on state."""
        # Base strategy: random movement
        return np.random.uniform(-1, 1, 2)
        
    def reset(self):
        """Reset strategy state."""
        pass
        
    def _add_noise(self, action: np.ndarray) -> np.ndarray:
        """Add noise to action based on difficulty level."""
        # Higher difficulty = less noise
        noise_scale = max(0, 1.0 - self.difficulty) * 0.5
        noise = np.random.normal(0, noise_scale, 2)
        action = action + noise
        
        # Normalize if needed
        if np.linalg.norm(action) > 1.0:
            action = action / np.linalg.norm(action)
            
        return action

class RandomStrategy(OpponentStrategy):
    """Random movement strategy with slight bias toward objectives."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "Random"
        
    def get_action(self, agent: Agent, state: Dict[str, Any]) -> np.ndarray:
        """Get random action with occasional bias toward objectives."""
        # Base random movement
        action = np.random.uniform(-1, 1, 2)
        
        # Occasionally bias toward objectives based on difficulty
        if random.random() < self.difficulty * 0.3:
            if agent.has_flag:
                # Return to base with flag
                base_pos = next((b for team, b in state.get('team_bases', {}).items() 
                               if team == agent.team), None)
                if base_pos is not None:
                    direction = np.array(base_pos) - np.array(agent.position)
                    if np.linalg.norm(direction) > 0:
                        action = direction / np.linalg.norm(direction)
            else:
                # Go for enemy flag
                enemy_flags = [f for f in state.get('flags', []) 
                              if f.get('team') != agent.team and not f.get('is_captured', False)]
                if enemy_flags:
                    flag = random.choice(enemy_flags)
                    direction = np.array(flag.get('position')) - np.array(agent.position)
                    if np.linalg.norm(direction) > 0:
                        action = direction / np.linalg.norm(direction)
        
        return self._add_noise(action)

class DirectStrategy(OpponentStrategy):
    """Direct path to objective strategy."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "Direct"
        
    def get_action(self, agent: Agent, state: Dict[str, Any]) -> np.ndarray:
        """Move directly toward objectives."""
        if agent.is_tagged:
            # Can't move if tagged
            return np.zeros(2)
            
        if agent.has_flag:
            # Return to base with flag
            base_pos = next((b for team, b in state.get('team_bases', {}).items() 
                           if team == agent.team), None)
            if base_pos is not None:
                direction = np.array(base_pos) - np.array(agent.position)
                if np.linalg.norm(direction) > 0:
                    action = direction / np.linalg.norm(direction)
                    return self._add_noise(action)
        else:
            # Go for enemy flag
            enemy_flags = [f for f in state.get('flags', []) 
                          if f.get('team') != agent.team and not f.get('is_captured', False)]
            if enemy_flags:
                closest_flag = min(enemy_flags, 
                                  key=lambda f: np.linalg.norm(np.array(f.get('position')) - np.array(agent.position)))
                direction = np.array(closest_flag.get('position')) - np.array(agent.position)
                if np.linalg.norm(direction) > 0:
                    action = direction / np.linalg.norm(direction)
                    return self._add_noise(action)
        
        # Default random movement if no objective
        return np.random.uniform(-1, 1, 2)

class DefensiveStrategy(OpponentStrategy):
    """Focus on defending own flag."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "Defensive"
        self.patrol_radius = config.get('patrol_radius', 20.0)
        self.patrol_target = None
        self.patrol_timer = 0
        
    def reset(self):
        """Reset patrol targets."""
        self.patrol_target = None
        self.patrol_timer = 0
        
    def get_action(self, agent: Agent, state: Dict[str, Any]) -> np.ndarray:
        """Defend own flag and base."""
        if agent.is_tagged:
            # Can't move if tagged
            return np.zeros(2)
            
        # Find own flag
        own_flag = next((f for f in state.get('flags', []) 
                       if f.get('team') == agent.team), None)
        
        if own_flag and not own_flag.get('is_captured', False):
            flag_pos = np.array(own_flag.get('position'))
            agent_pos = np.array(agent.position)
            dist_to_flag = np.linalg.norm(flag_pos - agent_pos)
            
            # If enemy is near flag, intercept
            enemy_agents = [a for a in state.get('agents', []) 
                          if a.get('team') != agent.team and not a.get('is_tagged', False)]
            
            closest_enemy = None
            min_dist_to_flag = float('inf')
            
            for enemy in enemy_agents:
                enemy_pos = np.array(enemy.get('position'))
                enemy_dist_to_flag = np.linalg.norm(enemy_pos - flag_pos)
                if enemy_dist_to_flag < min_dist_to_flag:
                    min_dist_to_flag = enemy_dist_to_flag
                    closest_enemy = enemy
            
            # If enemy is near and approaching flag, intercept
            if closest_enemy and min_dist_to_flag < self.patrol_radius * 1.5:
                enemy_pos = np.array(closest_enemy.get('position'))
                # Intercept by heading to position between enemy and flag
                intercept_pos = flag_pos + 0.7 * (enemy_pos - flag_pos)
                direction = intercept_pos - agent_pos
                if np.linalg.norm(direction) > 0:
                    action = direction / np.linalg.norm(direction)
                    return self._add_noise(action)
            
            # Patrol around flag
            self.patrol_timer -= 1
            if self.patrol_timer <= 0 or self.patrol_target is None:
                # Generate new patrol target
                angle = random.uniform(0, 2 * math.pi)
                offset = random.uniform(0.5, 1.0) * self.patrol_radius
                self.patrol_target = flag_pos + np.array([
                    math.cos(angle) * offset,
                    math.sin(angle) * offset
                ])
                self.patrol_timer = random.randint(20, 40)
            
            # Move to patrol target
            if self.patrol_target is not None:
                direction = self.patrol_target - agent_pos
                if np.linalg.norm(direction) > 0:
                    action = direction / np.linalg.norm(direction)
                    return self._add_noise(action)
        
        # Fallback: move toward own flag
        if own_flag:
            direction = np.array(own_flag.get('position')) - np.array(agent.position)
            if np.linalg.norm(direction) > 0:
                action = direction / np.linalg.norm(direction)
                return self._add_noise(action)
        
        # Default random movement
        return np.random.uniform(-1, 1, 2)

class AggressiveStrategy(OpponentStrategy):
    """Focus on tagging opponents."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "Aggressive"
        
    def get_action(self, agent: Agent, state: Dict[str, Any]) -> np.ndarray:
        """Aggressively pursue and tag enemy agents."""
        if agent.is_tagged:
            # Can't move if tagged
            return np.zeros(2)
            
        agent_pos = np.array(agent.position)
        
        # Prioritize enemy with flag
        enemy_agents = [a for a in state.get('agents', []) 
                      if a.get('team') != agent.team and not a.get('is_tagged', False)]
        
        # First prioritize enemies with flag
        flag_carriers = [e for e in enemy_agents if e.get('has_flag', False)]
        if flag_carriers:
            target = min(flag_carriers, 
                       key=lambda e: np.linalg.norm(np.array(e.get('position')) - agent_pos))
            direction = np.array(target.get('position')) - agent_pos
            if np.linalg.norm(direction) > 0:
                action = direction / np.linalg.norm(direction)
                return self._add_noise(action)
        
        # Next prioritize closest enemy
        if enemy_agents:
            target = min(enemy_agents, 
                       key=lambda e: np.linalg.norm(np.array(e.get('position')) - agent_pos))
            direction = np.array(target.get('position')) - agent_pos
            if np.linalg.norm(direction) > 0:
                action = direction / np.linalg.norm(direction)
                return self._add_noise(action)
        
        # If no visible enemies, move toward enemy territory
        enemy_base = next((b for team, b in state.get('team_bases', {}).items() 
                         if team != agent.team), None)
        if enemy_base is not None:
            direction = np.array(enemy_base) - agent_pos
            if np.linalg.norm(direction) > 0:
                action = direction / np.linalg.norm(direction)
                return self._add_noise(action)
        
        # Default random movement
        return np.random.uniform(-1, 1, 2)

class CoordinatedStrategy(OpponentStrategy):
    """Team-based strategy with role assignments."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "Coordinated"
        self.roles = {}  # agent_id -> role
        self.target_positions = {}  # agent_id -> target position
        self.target_timers = {}  # agent_id -> timer for target recalculation
        
        # Role distribution
        role_dist = config.get('role_distribution', {
            'attacker': 0.4,
            'defender': 0.3,
            'interceptor': 0.3
        })
        self.role_distribution = role_dist
        
    def reset(self):
        """Reset team coordination state."""
        self.roles = {}
        self.target_positions = {}
        self.target_timers = {}
        
    def assign_roles(self, agents: List[Dict[str, Any]]):
        """Assign roles to team members."""
        # Only process opponent team agents
        team_agents = [a for a in agents if a.get('id') not in self.roles]
        if not team_agents:
            return
            
        # Determine how many of each role based on distribution
        num_agents = len(team_agents)
        roles_to_assign = []
        
        for role, percentage in self.role_distribution.items():
            count = max(1, round(num_agents * percentage))
            roles_to_assign.extend([role] * count)
            
        # Ensure we have exactly the right number of roles
        while len(roles_to_assign) < num_agents:
            roles_to_assign.append('attacker')
        roles_to_assign = roles_to_assign[:num_agents]
        
        # Randomly assign roles
        random.shuffle(roles_to_assign)
        
        for i, agent in enumerate(team_agents):
            self.roles[agent.get('id')] = roles_to_assign[i]
    
    def get_action(self, agent: Agent, state: Dict[str, Any]) -> np.ndarray:
        """Get coordinated action based on agent's role."""
        if agent.is_tagged:
            # Can't move if tagged
            return np.zeros(2)
        
        # Initialize roles if needed
        if not self.roles:
            self.assign_roles(state.get('agents', []))
            
        # Get agent's role
        role = self.roles.get(agent.id, 'attacker')
        
        # Handle different roles
        if role == 'attacker':
            return self._get_attacker_action(agent, state)
        elif role == 'defender':
            return self._get_defender_action(agent, state)
        elif role == 'interceptor':
            return self._get_interceptor_action(agent, state)
        else:
            # Default to attacker
            return self._get_attacker_action(agent, state)
    
    def _get_attacker_action(self, agent: Agent, state: Dict[str, Any]) -> np.ndarray:
        """Focus on capturing enemy flag."""
        agent_pos = np.array(agent.position)
        
        if agent.has_flag:
            # Return to base with flag
            base_pos = next((b for team, b in state.get('team_bases', {}).items() 
                           if team == agent.team), None)
            if base_pos is not None:
                direction = np.array(base_pos) - agent_pos
                if np.linalg.norm(direction) > 0:
                    action = direction / np.linalg.norm(direction)
                    return self._add_noise(action)
        else:
            # Go for enemy flag using smarter path
            enemy_flags = [f for f in state.get('flags', []) 
                          if f.get('team') != agent.team and not f.get('is_captured', False)]
            
            if enemy_flags:
                closest_flag = min(enemy_flags, 
                                  key=lambda f: np.linalg.norm(np.array(f.get('position')) - agent_pos))
                flag_pos = np.array(closest_flag.get('position'))
                
                # Check if path to flag is clear
                enemy_agents = [a for a in state.get('agents', []) 
                               if a.get('team') != agent.team and not a.get('is_tagged', False)]
                
                # Find potential threats along path
                threats = []
                for enemy in enemy_agents:
                    enemy_pos = np.array(enemy.get('position'))
                    # Check if enemy is along path to flag
                    path_vector = flag_pos - agent_pos
                    path_length = np.linalg.norm(path_vector)
                    if path_length > 0:
                        path_dir = path_vector / path_length
                        enemy_vector = enemy_pos - agent_pos
                        projection = np.dot(enemy_vector, path_dir)
                        
                        # Only consider enemies ahead of us
                        if 0 < projection < path_length:
                            # Calculate perpendicular distance to path
                            perp_dist = np.linalg.norm(enemy_vector - projection * path_dir)
                            if perp_dist < 15.0:  # Threat threshold
                                threats.append((enemy, projection, perp_dist))
                
                # If threats exist, try to avoid them
                if threats:
                    # Sort by threat level (combination of distance and projection)
                    threats.sort(key=lambda t: t[1] * (1.0 - t[2] / 15.0))
                    main_threat = threats[0][0]
                    
                    # Decide if we should evade
                    if random.random() < self.difficulty * 0.7:
                        threat_pos = np.array(main_threat.get('position'))
                        
                        # Calculate evasion direction (perpendicular to threat)
                        threat_dir = threat_pos - agent_pos
                        if np.linalg.norm(threat_dir) > 0:
                            threat_dir = threat_dir / np.linalg.norm(threat_dir)
                            
                            # Calculate perpendicular vector (randomly left or right)
                            perp_dir = np.array([-threat_dir[1], threat_dir[0]])
                            if random.random() < 0.5:
                                perp_dir = -perp_dir
                                
                            # Mix evasion with goal direction
                            goal_dir = flag_pos - agent_pos
                            if np.linalg.norm(goal_dir) > 0:
                                goal_dir = goal_dir / np.linalg.norm(goal_dir)
                                
                            # More evasion than goal when threat is close
                            threat_dist = np.linalg.norm(threat_pos - agent_pos)
                            evasion_weight = max(0.0, min(1.0, 20.0 / threat_dist))
                            
                            action = (1.0 - evasion_weight) * goal_dir + evasion_weight * perp_dir
                            if np.linalg.norm(action) > 0:
                                action = action / np.linalg.norm(action)
                                return self._add_noise(action)
                
                # No threats or decided not to evade, go directly for flag
                direction = flag_pos - agent_pos
                if np.linalg.norm(direction) > 0:
                    action = direction / np.linalg.norm(direction)
                    return self._add_noise(action)
        
        # Default random movement
        return np.random.uniform(-1, 1, 2)
    
    def _get_defender_action(self, agent: Agent, state: Dict[str, Any]) -> np.ndarray:
        """Defend team flag and territory."""
        agent_pos = np.array(agent.position)
        
        # Find own flag
        own_flag = next((f for f in state.get('flags', []) 
                        if f.get('team') == agent.team), None)
        
        if own_flag:
            flag_pos = np.array(own_flag.get('position'))
            base_pos = next((b for team, b in state.get('team_bases', {}).items() 
                            if team == agent.team), None)
            
            # If flag is captured, intercept the carrier
            if own_flag.get('is_captured', False):
                carrier_id = own_flag.get('carrier_id')
                carrier = next((a for a in state.get('agents', []) 
                               if a.get('id') == carrier_id), None)
                
                if carrier:
                    carrier_pos = np.array(carrier.get('position'))
                    
                    # Try to intercept at point between carrier and base
                    if base_pos is not None:
                        base_pos = np.array(base_pos)
                        intercept_pos = carrier_pos + 0.6 * (base_pos - carrier_pos)
                        
                        direction = intercept_pos - agent_pos
                        if np.linalg.norm(direction) > 0:
                            action = direction / np.linalg.norm(direction)
                            return self._add_noise(action)
            
            # If flag not captured, defend it
            else:
                # Recalculate target position periodically
                if (agent.id not in self.target_timers or 
                    self.target_timers[agent.id] <= 0):
                    
                    # Position between flag and enemy base for interception
                    enemy_base = next((b for team, b in state.get('team_bases', {}).items() 
                                     if team != agent.team), None)
                    
                    if enemy_base is not None:
                        enemy_base = np.array(enemy_base)
                        # Calculate defensive position
                        direction_to_enemy = enemy_base - flag_pos
                        if np.linalg.norm(direction_to_enemy) > 0:
                            direction_to_enemy = direction_to_enemy / np.linalg.norm(direction_to_enemy)
                            
                            # Position between flag and enemy base
                            optimal_distance = random.uniform(10.0, 15.0)
                            target_pos = flag_pos + direction_to_enemy * optimal_distance
                            
                            # Store target position
                            self.target_positions[agent.id] = target_pos
                            self.target_timers[agent.id] = random.randint(30, 50)
                
                # Move toward target position
                if agent.id in self.target_positions:
                    target_pos = self.target_positions[agent.id]
                    direction = target_pos - agent_pos
                    
                    # If reached target, patrol around it
                    if np.linalg.norm(direction) < 5.0:
                        # Rotate around flag
                        angle = math.atan2(agent_pos[1] - flag_pos[1], 
                                         agent_pos[0] - flag_pos[0])
                        angle += random.uniform(0.1, 0.3)  # Slow rotation
                        
                        radius = np.linalg.norm(agent_pos - flag_pos)
                        target_pos = flag_pos + np.array([
                            math.cos(angle) * radius,
                            math.sin(angle) * radius
                        ])
                        direction = target_pos - agent_pos
                    
                    if np.linalg.norm(direction) > 0:
                        action = direction / np.linalg.norm(direction)
                        return self._add_noise(action)
                
                # Fallback: move toward flag
                direction = flag_pos - agent_pos
                if np.linalg.norm(direction) > 0:
                    action = direction / np.linalg.norm(direction)
                    return self._add_noise(action)
        
        # Default random movement
        return np.random.uniform(-1, 1, 2)
    
    def _get_interceptor_action(self, agent: Agent, state: Dict[str, Any]) -> np.ndarray:
        """Intercept enemy flag carriers and intruders."""
        agent_pos = np.array(agent.position)
        
        # Check if any opponent has our flag
        enemy_with_flag = next((a for a in state.get('agents', []) 
                              if a.get('team') != agent.team and 
                                a.get('has_flag', False) and 
                                not a.get('is_tagged', False)), None)
        
        if enemy_with_flag:
            # High priority: intercept enemy with flag
            enemy_pos = np.array(enemy_with_flag.get('position'))
            
            # Try to predict where enemy is going (usually toward their base)
            enemy_base = next((b for team, b in state.get('team_bases', {}).items() 
                             if team != agent.team), None)
            
            if enemy_base is not None:
                enemy_base = np.array(enemy_base)
                enemy_vel = np.array(enemy_with_flag.get('velocity', [0, 0]))
                
                # Predict position based on velocity and direction to base
                direction_to_base = enemy_base - enemy_pos
                if np.linalg.norm(direction_to_base) > 0:
                    direction_to_base = direction_to_base / np.linalg.norm(direction_to_base)
                
                # Combine current velocity with direction to base
                if np.linalg.norm(enemy_vel) > 0:
                    enemy_vel = enemy_vel / np.linalg.norm(enemy_vel)
                    predicted_dir = 0.7 * enemy_vel + 0.3 * direction_to_base
                else:
                    predicted_dir = direction_to_base
                
                if np.linalg.norm(predicted_dir) > 0:
                    predicted_dir = predicted_dir / np.linalg.norm(predicted_dir)
                
                # Calculate interception point
                distance = np.linalg.norm(enemy_pos - agent_pos)
                intercept_pos = enemy_pos + predicted_dir * (distance * 0.5)
                
                direction = intercept_pos - agent_pos
                if np.linalg.norm(direction) > 0:
                    action = direction / np.linalg.norm(direction)
                    return self._add_noise(action)
        
        # Check for enemies in our territory
        own_base = next((b for team, b in state.get('team_bases', {}).items() 
                       if team == agent.team), None)
        
        if own_base is not None:
            own_base = np.array(own_base)
            
            # Find enemies in our territory
            enemies_in_territory = []
            for enemy in [a for a in state.get('agents', []) 
                         if a.get('team') != agent.team and 
                           not a.get('is_tagged', False)]:
                
                enemy_pos = np.array(enemy.get('position'))
                
                # Simple territory check: distance to base
                territory_radius = 40.0
                if np.linalg.norm(enemy_pos - own_base) < territory_radius:
                    # Calculate threat level based on position and status
                    threat_level = 1.0 - (np.linalg.norm(enemy_pos - own_base) / territory_radius)
                    
                    # Higher threat if close to flag
                    own_flag = next((f for f in state.get('flags', []) 
                                   if f.get('team') == agent.team and 
                                     not f.get('is_captured', False)), None)
                    if own_flag:
                        flag_pos = np.array(own_flag.get('position'))
                        dist_to_flag = np.linalg.norm(enemy_pos - flag_pos)
                        flag_threat = max(0, 1.0 - dist_to_flag / 30.0)
                        threat_level = max(threat_level, flag_threat)
                    
                    enemies_in_territory.append((enemy, threat_level))
            
            if enemies_in_territory:
                # Sort by threat level
                enemies_in_territory.sort(key=lambda e: e[1], reverse=True)
                target_enemy = enemies_in_territory[0][0]
                
                enemy_pos = np.array(target_enemy.get('position'))
                direction = enemy_pos - agent_pos
                
                if np.linalg.norm(direction) > 0:
                    action = direction / np.linalg.norm(direction)
                    return self._add_noise(action)
        
        # Patrol territory or move to strategic position
        # Recalculate target position periodically
        if agent.id not in self.target_timers or self.target_timers[agent.id] <= 0:
            # Choose patrol point in our territory
            if own_base is not None:
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(20.0, 40.0)
                target_pos = own_base + np.array([
                    math.cos(angle) * radius,
                    math.sin(angle) * radius
                ])
                
                # Store target position
                self.target_positions[agent.id] = target_pos
                self.target_timers[agent.id] = random.randint(40, 60)
        
        # Decrease timer
        if agent.id in self.target_timers:
            self.target_timers[agent.id] -= 1
            
        # Move toward target position
        if agent.id in self.target_positions:
            target_pos = self.target_positions[agent.id]
            direction = target_pos - agent_pos
            
            if np.linalg.norm(direction) > 0:
                action = direction / np.linalg.norm(direction)
                return self._add_noise(action)
        
        # Default random movement
        return np.random.uniform(-1, 1, 2)

# Dictionary of available opponent strategies
OPPONENT_STRATEGIES = {
    "random": RandomStrategy,
    "direct": DirectStrategy,
    "defensive": DefensiveStrategy,
    "aggressive": AggressiveStrategy,
    "coordinated": CoordinatedStrategy
}

def get_opponent_strategy(name: str, config: Dict[str, Any] = None) -> OpponentStrategy:
    """Get opponent strategy by name."""
    if name not in OPPONENT_STRATEGIES:
        print(f"Warning: Strategy '{name}' not found. Using 'random' instead.")
        name = "random"
        
    return OPPONENT_STRATEGIES[name](config) 