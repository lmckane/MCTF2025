import numpy as np
from typing import Dict, Any, List
from hrl.environment.game_env import Agent, Flag, GameState

class OpponentStrategy:
    """Base class for opponent strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize strategy with config."""
        self.config = config or {}
        self.name = "Base"
        
    def get_action(self, agent: Agent, state: Dict[str, Any]) -> np.ndarray:
        """Get action for agent based on state."""
        # Base strategy: random movement
        return np.random.uniform(-1, 1, 2)
        
    def reset(self):
        """Reset strategy state."""
        pass

class RandomStrategy(OpponentStrategy):
    """Random movement strategy."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "Random"
        
    def get_action(self, agent: Agent, state: Dict[str, Any]) -> np.ndarray:
        # Simply move in random directions
        return np.random.uniform(-1, 1, 2)

class DirectStrategy(OpponentStrategy):
    """Direct flag capture strategy."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "Direct"
        
    def get_action(self, agent: Agent, state: Dict[str, Any]) -> np.ndarray:
        # Extract game state elements
        agents = state['agents']
        flags = state['flags']
        team_bases = state['team_bases']
        
        # If agent is tagged, we can't move
        if agent.is_tagged:
            return np.zeros(2)
            
        # If agent has a flag, go back to base
        if agent.has_flag:
            base_pos = team_bases[agent.team]
            direction = base_pos - agent.position
        else:
            # Go for enemy flags
            enemy_flags = [flag for flag in flags if flag.team != agent.team and not flag.is_captured]
            if enemy_flags:
                # Find closest flag
                closest_flag = min(enemy_flags, key=lambda f: np.linalg.norm(agent.position - f.position))
                direction = closest_flag.position - agent.position
            else:
                # All enemy flags captured, go to base
                base_pos = team_bases[agent.team]
                direction = base_pos - agent.position
                
        # Normalize direction
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
            
        return direction
        
class DefensiveStrategy(OpponentStrategy):
    """Flag defense strategy."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "Defensive"
        
    def get_action(self, agent: Agent, state: Dict[str, Any]) -> np.ndarray:
        # Extract game state elements
        agents = state['agents']
        flags = state['flags']
        team_bases = state['team_bases']
        
        # If agent is tagged, we can't move
        if agent.is_tagged:
            return np.zeros(2)
            
        # Find our team's flag
        our_flag = next((flag for flag in flags if flag.team == agent.team), None)
        
        # If our flag is captured, hunt the carrier
        if our_flag and our_flag.is_captured:
            carrier = next((a for a in agents if a.id == our_flag.carrier_id), None)
            if carrier:
                direction = carrier.position - agent.position
            else:
                # Flag in transport but carrier not found
                base_pos = team_bases[agent.team]
                direction = base_pos - agent.position
        else:
            # Guard our flag
            if our_flag:
                # Position between base and center
                base_pos = team_bases[agent.team]
                center = np.array(state.get('map_size', [100, 100])) / 2
                target_pos = base_pos + 0.3 * (center - base_pos)
                direction = target_pos - agent.position
            else:
                # No flag to guard, move randomly
                direction = np.random.uniform(-1, 1, 2)
                
        # Normalize direction
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
            
        return direction

class AggressiveStrategy(OpponentStrategy):
    """Aggressive tagging strategy."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "Aggressive"
        
    def get_action(self, agent: Agent, state: Dict[str, Any]) -> np.ndarray:
        # Extract game state elements
        agents = state['agents']
        flags = state['flags']
        team_bases = state['team_bases']
        territories = state['territories']
        
        # If agent is tagged, we can't move
        if agent.is_tagged:
            return np.zeros(2)
            
        # Find enemy agents in our territory
        enemy_agents = [a for a in agents if a.team != agent.team and not a.is_tagged]
        
        # Find enemy agents with flags
        flag_carriers = [a for a in enemy_agents if a.has_flag]
        
        if flag_carriers:
            # Prioritize chasing flag carriers
            closest_carrier = min(flag_carriers, key=lambda a: np.linalg.norm(agent.position - a.position))
            direction = closest_carrier.position - agent.position
        elif enemy_agents:
            # Chase any enemy in range
            closest_enemy = min(enemy_agents, key=lambda a: np.linalg.norm(agent.position - a.position))
            direction = closest_enemy.position - agent.position
        else:
            # No enemies in range, patrol our territory
            if agent.has_flag:
                # Return to base with flag
                base_pos = team_bases[agent.team]
                direction = base_pos - agent.position
            else:
                # Go for enemy flags
                enemy_flags = [flag for flag in flags if flag.team != agent.team and not flag.is_captured]
                if enemy_flags:
                    closest_flag = min(enemy_flags, key=lambda f: np.linalg.norm(agent.position - f.position))
                    direction = closest_flag.position - agent.position
                else:
                    # Patrol around our base
                    base_pos = team_bases[agent.team]
                    angle = np.random.uniform(0, 2*np.pi)
                    patrol_point = base_pos + np.array([15*np.cos(angle), 15*np.sin(angle)])
                    direction = patrol_point - agent.position
                    
        # Normalize direction
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
            
        return direction

class CoordinatedStrategy(OpponentStrategy):
    """Team-based coordinated strategy."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "Coordinated"
        self.roles = {}  # Assign roles to agents
        self.roles_initialized = False
        
    def reset(self):
        """Reset strategy state."""
        self.roles_initialized = False
        self.roles = {}
        
    def get_action(self, agent: Agent, state: Dict[str, Any]) -> np.ndarray:
        # Initialize roles if needed
        if not self.roles_initialized:
            self._initialize_roles(state)
            
        # Extract agent properties
        agent_id = agent.id if hasattr(agent, 'id') else agent['id']
        agent_team = agent.team if hasattr(agent, 'team') else agent['team']
        agent_position = agent.position if hasattr(agent, 'position') else np.array(agent['position'])
        agent_has_flag = agent.has_flag if hasattr(agent, 'has_flag') else agent['has_flag']
        agent_is_tagged = agent.is_tagged if hasattr(agent, 'is_tagged') else agent['is_tagged']
            
        # Get role for this agent
        role = self.roles.get(agent_id, "attacker")
        
        # Extract game state elements
        agents = state['agents']
        flags = state['flags']
        team_bases = state['team_bases']
        
        # If agent is tagged, we can't move
        if agent_is_tagged:
            return np.zeros(2)
            
        # Defender behavior
        if role == "defender":
            # Get our team's flag and base
            our_flag = next((f for f in flags if (hasattr(f, 'team') and f.team == agent_team) or 
                             (isinstance(f, dict) and f['team'] == agent_team)), None)
            base_pos = team_bases[agent_team]
            
            our_flag_is_captured = our_flag.is_captured if hasattr(our_flag, 'is_captured') else our_flag['is_captured']
            our_flag_carrier_id = our_flag.carrier_id if hasattr(our_flag, 'carrier_id') else our_flag.get('carrier_id')
            
            if our_flag and our_flag_is_captured:
                # Chase the carrier
                carrier = next((a for a in agents if (hasattr(a, 'id') and a.id == our_flag_carrier_id) or 
                                (isinstance(a, dict) and a['id'] == our_flag_carrier_id)), None)
                if carrier:
                    carrier_position = carrier.position if hasattr(carrier, 'position') else np.array(carrier['position'])
                    direction = carrier_position - agent_position
                else:
                    # Patrol base
                    angle = (agent_id * 2.5) % (2*np.pi)  # Different angle for each defender
                    patrol_radius = 15
                    patrol_point = base_pos + np.array([patrol_radius*np.cos(angle), patrol_radius*np.sin(angle)])
                    direction = patrol_point - agent_position
            else:
                # Patrol base
                angle = (agent_id * 2.5) % (2*np.pi)  # Different angle for each defender
                patrol_radius = 15
                patrol_point = base_pos + np.array([patrol_radius*np.cos(angle), patrol_radius*np.sin(angle)])
                direction = patrol_point - agent_position
                
        # Attacker behavior
        elif role == "attacker":
            if agent_has_flag:
                # Return to base with flag
                base_pos = team_bases[agent_team]
                direction = base_pos - agent_position
            else:
                # Go for enemy flags
                enemy_flags = [f for f in flags if ((hasattr(f, 'team') and f.team != agent_team) or 
                                                    (isinstance(f, dict) and f['team'] != agent_team)) and
                                                   ((hasattr(f, 'is_captured') and not f.is_captured) or 
                                                    (isinstance(f, dict) and not f['is_captured']))]
                if enemy_flags:
                    # Assign different flags to different attackers if multiple flags
                    if len(enemy_flags) > 1:
                        flag_index = agent_id % len(enemy_flags)
                        target_flag = enemy_flags[flag_index]
                    else:
                        target_flag = enemy_flags[0]
                        
                    flag_position = target_flag.position if hasattr(target_flag, 'position') else np.array(target_flag['position'])
                    direction = flag_position - agent_position
                else:
                    # Hunt enemy players
                    enemy_agents = [a for a in agents if ((hasattr(a, 'team') and a.team != agent_team) or 
                                                          (isinstance(a, dict) and a['team'] != agent_team)) and
                                                         ((hasattr(a, 'is_tagged') and not a.is_tagged) or 
                                                          (isinstance(a, dict) and not a['is_tagged']))]
                    if enemy_agents:
                        # Find closest enemy agent
                        def get_distance(a):
                            a_pos = a.position if hasattr(a, 'position') else np.array(a['position'])
                            return np.linalg.norm(agent_position - a_pos)
                        
                        closest_enemy = min(enemy_agents, key=get_distance)
                        enemy_position = closest_enemy.position if hasattr(closest_enemy, 'position') else np.array(closest_enemy['position'])
                        direction = enemy_position - agent_position
                    else:
                        # Return to base
                        base_pos = team_bases[agent_team]
                        direction = base_pos - agent_position
                        
        # Normalize direction
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
            
        # Add some noise for less predictable movement
        noise = np.random.normal(0, 0.1, 2)
        direction = np.clip(direction + noise, -1, 1)
            
        return direction
        
    def _initialize_roles(self, state):
        """Assign roles to team members."""
        agents = state['agents']
        team_agents = {}
        
        # Group agents by team
        for agent in agents:
            # Handle both Agent objects and dictionary representations
            team = agent['team'] if isinstance(agent, dict) else agent.team
            agent_id = agent['id'] if isinstance(agent, dict) else agent.id
            
            if team not in team_agents:
                team_agents[team] = []
            team_agents[team].append({'agent': agent, 'id': agent_id})
            
        # Assign roles within each team
        for team, team_agents_list in team_agents.items():
            if len(team_agents_list) >= 3:
                # 1 defender, 2 attackers for 3+ agents
                for i, agent_data in enumerate(team_agents_list):
                    if i == 0:
                        self.roles[agent_data['id']] = "defender"
                    else:
                        self.roles[agent_data['id']] = "attacker"
            else:
                # All attackers for small teams
                for agent_data in team_agents_list:
                    self.roles[agent_data['id']] = "attacker"
                    
        self.roles_initialized = True

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