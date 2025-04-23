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
            
        # Get role for this agent
        role = self.roles.get(agent.id, "attacker")
        
        # Extract game state elements
        agents = state['agents']
        flags = state['flags']
        team_bases = state['team_bases']
        
        # If agent is tagged, we can't move
        if agent.is_tagged:
            return np.zeros(2)
            
        # Defender behavior
        if role == "defender":
            # Get our team's flag and base
            our_flag = next((flag for flag in flags if flag.team == agent.team), None)
            base_pos = team_bases[agent.team]
            
            if our_flag and our_flag.is_captured:
                # Chase the carrier
                carrier = next((a for a in agents if a.id == our_flag.carrier_id), None)
                if carrier:
                    direction = carrier.position - agent.position
                else:
                    # Patrol base
                    angle = (agent.id * 2.5) % (2*np.pi)  # Different angle for each defender
                    patrol_radius = 15
                    patrol_point = base_pos + np.array([patrol_radius*np.cos(angle), patrol_radius*np.sin(angle)])
                    direction = patrol_point - agent.position
            else:
                # Patrol base
                angle = (agent.id * 2.5) % (2*np.pi)  # Different angle for each defender
                patrol_radius = 15
                patrol_point = base_pos + np.array([patrol_radius*np.cos(angle), patrol_radius*np.sin(angle)])
                direction = patrol_point - agent.position
                
        # Attacker behavior
        elif role == "attacker":
            if agent.has_flag:
                # Return to base with flag
                base_pos = team_bases[agent.team]
                direction = base_pos - agent.position
            else:
                # Go for enemy flags
                enemy_flags = [flag for flag in flags if flag.team != agent.team and not flag.is_captured]
                if enemy_flags:
                    # Assign different flags to different attackers if multiple flags
                    if len(enemy_flags) > 1:
                        flag_index = agent.id % len(enemy_flags)
                        target_flag = enemy_flags[flag_index]
                    else:
                        target_flag = enemy_flags[0]
                    direction = target_flag.position - agent.position
                else:
                    # Hunt enemy players
                    enemy_agents = [a for a in agents if a.team != agent.team and not a.is_tagged]
                    if enemy_agents:
                        closest_enemy = min(enemy_agents, key=lambda a: np.linalg.norm(agent.position - a.position))
                        direction = closest_enemy.position - agent.position
                    else:
                        # Return to base
                        base_pos = team_bases[agent.team]
                        direction = base_pos - agent.position
                        
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
            if agent.team not in team_agents:
                team_agents[agent.team] = []
            team_agents[agent.team].append(agent)
            
        # Assign roles within each team
        for team, agents in team_agents.items():
            if len(agents) >= 3:
                # 1 defender, 2 attackers for 3+ agents
                for i, agent in enumerate(agents):
                    if i == 0:
                        self.roles[agent.id] = "defender"
                    else:
                        self.roles[agent.id] = "attacker"
            else:
                # All attackers for small teams
                for agent in agents:
                    self.roles[agent.id] = "attacker"
                    
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