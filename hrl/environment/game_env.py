import numpy as np
from typing import Dict, Any, Tuple, List
import random
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import torch

class GameState(Enum):
    """Possible game states."""
    PLAYING = 0
    WON = 1
    LOST = 2
    DRAW = 3

@dataclass
class Agent:
    """Represents an agent in the game."""
    position: np.ndarray
    velocity: np.ndarray
    has_flag: bool
    is_tagged: bool
    team: int
    health: float = 100.0
    last_action: np.ndarray = None

@dataclass
class Flag:
    """Represents a flag in the game."""
    position: np.ndarray
    is_captured: bool
    team: int
    carrier_id: int = None

class GameEnvironment:
    """Environment for the capture-the-flag game."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the game environment.
        
        Args:
            config: Configuration dictionary containing:
                - map_size: Size of the game map (default: [100, 100])
                - num_agents: Number of agents per team (default: 3)
                - max_steps: Maximum steps per episode (default: 1000)
                - tag_radius: Radius for tagging (default: 5)
                - capture_radius: Radius for capturing flag (default: 10)
                - base_radius: Radius of team bases (default: 20)
                - difficulty: Current difficulty level (default: 0.5)
        """
        self.config = config
        self.map_size = np.array(config.get('map_size', [100, 100]))
        self.num_agents = config.get('num_agents', 3)
        self.max_steps = config.get('max_steps', 1000)
        self.tag_radius = config.get('tag_radius', 5)
        self.capture_radius = config.get('capture_radius', 10)
        self.base_radius = config.get('base_radius', 20)
        self.difficulty = config.get('difficulty', 0.5)
        
        # Initialize game state
        self.agents: List[Agent] = []
        self.flags: List[Flag] = []
        self.team_bases: Dict[int, np.ndarray] = {}
        self.territories: Dict[int, List[np.ndarray]] = {}
        self.step_count = 0
        self.game_state = GameState.PLAYING
        
        # Set up the game
        self._setup_game()
        
    def _setup_game(self):
        """Set up the initial game state."""
        # Clear existing state
        self.agents.clear()
        self.flags.clear()
        self.team_bases.clear()
        self.territories.clear()
        
        # Place team bases
        self.team_bases[0] = np.array([self.base_radius, self.map_size[1]/2])
        self.team_bases[1] = np.array([self.map_size[0] - self.base_radius, self.map_size[1]/2])
        
        # Place flags
        self.flags.append(Flag(
            position=self.team_bases[0] + np.array([10, 0]),
            is_captured=False,
            team=0
        ))
        self.flags.append(Flag(
            position=self.team_bases[1] - np.array([10, 0]),
            is_captured=False,
            team=1
        ))
        
        # Place agents
        for team in [0, 1]:
            for i in range(self.num_agents):
                # Position agents near their base
                base_pos = self.team_bases[team]
                pos = base_pos + np.random.uniform(-10, 10, 2)
                pos = np.clip(pos, [0, 0], self.map_size)
                
                self.agents.append(Agent(
                    position=pos,
                    velocity=np.zeros(2),
                    has_flag=False,
                    is_tagged=False,
                    team=team
                ))
                
        # Define territories
        self.territories[0] = [
            np.array([0, 0]),
            np.array([self.map_size[0]/2, 0]),
            np.array([self.map_size[0]/2, self.map_size[1]]),
            np.array([0, self.map_size[1]])
        ]
        self.territories[1] = [
            np.array([self.map_size[0]/2, 0]),
            np.array([self.map_size[0], 0]),
            np.array([self.map_size[0], self.map_size[1]]),
            np.array([self.map_size[0]/2, self.map_size[1]])
        ]
        
    def reset(self) -> Dict[str, Any]:
        """Reset the environment to initial state."""
        self._setup_game()
        self.step_count = 0
        self.game_state = GameState.PLAYING
        return self._get_observation()
        
    def step(self, actions: List[np.ndarray]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            actions: List of actions for each agent. Each action is a 2D array [dx, dy].
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.game_state != GameState.PLAYING:
            return self._get_observation(), 0, True, {'game_state': self.game_state}
            
        self.step_count += 1
        
        # Update agent positions and velocities
        for i, (agent, action) in enumerate(zip(self.agents, actions)):
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            agent.last_action = action
            agent.velocity = action  # Action is directly velocity
            agent.position += agent.velocity
            
            # Clip position to map bounds
            agent.position = np.clip(agent.position, [0, 0], self.map_size)
            
        # Handle tagging
        self._handle_tagging()
        
        # Handle flag captures
        self._handle_flag_captures()
        
        # Check game over conditions
        done = self._check_game_over()
        
        # Calculate rewards
        reward = self._calculate_rewards()
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, info
        
    def _handle_tagging(self):
        """Handle agent tagging logic."""
        for i, agent in enumerate(self.agents):
            if agent.is_tagged:
                continue
                
            for j, other in enumerate(self.agents):
                if other.team == agent.team or other.is_tagged:
                    continue
                    
                dist = np.linalg.norm(agent.position - other.position)
                if dist <= self.tag_radius:
                    other.is_tagged = True
                    if other.has_flag:
                        # Drop flag if tagged while carrying it
                        for flag in self.flags:
                            if flag.carrier_id == j:
                                flag.is_captured = False
                                flag.carrier_id = None
                                other.has_flag = False
                                
    def _handle_flag_captures(self):
        """Handle flag capture logic."""
        for flag in self.flags:
            if flag.is_captured:
                continue
                
            for i, agent in enumerate(self.agents):
                if agent.team == flag.team or agent.is_tagged:
                    continue
                    
                dist = np.linalg.norm(agent.position - flag.position)
                if dist <= self.capture_radius:
                    flag.is_captured = True
                    flag.carrier_id = i
                    agent.has_flag = True
                    
    def _check_game_over(self) -> bool:
        """Check if the game is over."""
        # Check step limit
        if self.step_count >= self.max_steps:
            self.game_state = GameState.DRAW
            return True
            
        # Check if any team has captured the opponent's flag
        for flag in self.flags:
            if flag.is_captured:
                carrier = self.agents[flag.carrier_id]
                if carrier.team != flag.team:  # Captured opponent's flag
                    self.game_state = GameState.WON if carrier.team == 0 else GameState.LOST
                    return True
                    
        return False
        
    def _calculate_rewards(self) -> float:
        """Calculate rewards for the current state."""
        reward = 0.0
        
        # Reward for capturing flag
        for flag in self.flags:
            if flag.is_captured and self.agents[flag.carrier_id].team != flag.team:
                reward += 100.0
                
        # Reward for tagging opponents
        for agent in self.agents:
            if agent.is_tagged:
                reward -= 10.0
                
        # Reward for being in opponent territory
        for agent in self.agents:
            if self._is_in_territory(agent.position, 1 - agent.team):
                reward += 1.0
                
        return reward
        
    def _is_in_territory(self, position: np.ndarray, team: int) -> bool:
        """Check if a position is in a team's territory using ray casting algorithm."""
        x, y = position
        territory = self.territories[team]
        n = len(territory)
        inside = False
        
        p1x, p1y = territory[0]
        for i in range(1, n + 1):
            p2x, p2y = territory[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
        
    def _get_observation(self) -> Dict[str, Any]:
        """Get the current observation."""
        return {
            'agents': [{
                'position': agent.position,
                'velocity': agent.velocity,
                'has_flag': agent.has_flag,
                'is_tagged': agent.is_tagged,
                'team': agent.team,
                'health': agent.health
            } for agent in self.agents],
            'flags': [{
                'position': flag.position,
                'is_captured': flag.is_captured,
                'team': flag.team,
                'carrier_id': flag.carrier_id
            } for flag in self.flags],
            'team_bases': self.team_bases,
            'territories': self.territories,
            'step_count': self.step_count,
            'game_state': self.game_state
        }
        
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state."""
        return {
            'game_state': self.game_state,
            'step_count': self.step_count,
            'flag_captured': any(flag.is_captured for flag in self.flags),
            'tagged': sum(1 for agent in self.agents if agent.is_tagged),
            'died': sum(1 for agent in self.agents if agent.health <= 0),
            'option_success': {}  # To be filled by the policy
        }
        
    def render(self, mode: str = 'human'):
        """Render the current state of the environment."""
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            self.ax.set_xlim(0, self.map_size[0])
            self.ax.set_ylim(0, self.map_size[1])
            self.ax.set_aspect('equal')
            
        # Clear previous frame
        self.ax.clear()
        
        # Draw territories
        for team, territory in self.territories.items():
            color = 'lightblue' if team == 0 else 'lightpink'
            self.ax.fill(*zip(*territory), color=color, alpha=0.3)
            
        # Draw team bases
        for team, base in self.team_bases.items():
            color = 'blue' if team == 0 else 'red'
            circle = patches.Circle(base, self.base_radius, color=color, alpha=0.5)
            self.ax.add_patch(circle)
            
        # Draw flags
        for flag in self.flags:
            color = 'blue' if flag.team == 0 else 'red'
            if flag.is_captured:
                # Draw flag at carrier's position
                carrier = self.agents[flag.carrier_id]
                self.ax.plot(carrier.position[0], carrier.position[1], 
                           marker='^', color=color, markersize=15)
            else:
                # Draw flag at its position
                self.ax.plot(flag.position[0], flag.position[1], 
                           marker='^', color=color, markersize=15)
                           
        # Draw agents
        for agent in self.agents:
            color = 'blue' if agent.team == 0 else 'red'
            marker = 'o' if not agent.is_tagged else 'x'
            size = 10 if not agent.has_flag else 15
            
            self.ax.plot(agent.position[0], agent.position[1], 
                        marker=marker, color=color, markersize=size)
                        
            # Draw velocity vector
            if agent.velocity is not None and np.any(agent.velocity != 0):
                self.ax.arrow(agent.position[0], agent.position[1],
                            agent.velocity[0], agent.velocity[1],
                            head_width=2, head_length=3, fc=color, ec=color)
                            
        # Add text information
        self.ax.text(5, self.map_size[1] - 5, 
                    f'Step: {self.step_count}/{self.max_steps}',
                    fontsize=12)
                    
        if self.game_state != GameState.PLAYING:
            result = {
                GameState.WON: 'Team 0 Won!',
                GameState.LOST: 'Team 1 Won!',
                GameState.DRAW: 'Draw!'
            }[self.game_state]
            self.ax.text(self.map_size[0]/2, self.map_size[1]/2,
                        result, fontsize=20, ha='center')
                        
        plt.pause(0.01)  # Small pause to allow the plot to update
        
    def close(self):
        """Close the rendering window."""
        if hasattr(self, 'fig'):
            plt.close(self.fig)
            del self.fig
            del self.ax 