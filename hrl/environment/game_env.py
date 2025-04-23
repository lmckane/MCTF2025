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
    id: int = 0
    health: float = 100.0
    last_action: np.ndarray = None
    tag_timer: int = 0

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
                - max_velocity: Maximum agent velocity (default: 5.0)
                - win_score: Score needed to win (default: 3)
                - debug_level: Level of debug output (0=none, 1=minimal, 2=verbose)
        """
        self.config = config
        self.map_size = np.array(config.get('map_size', [100, 100]))
        self.num_agents = config.get('num_agents', 3)
        self.max_steps = config.get('max_steps', 500)
        self.tag_radius = config.get('tag_radius', 5)
        self.capture_radius = config.get('capture_radius', 10)
        self.base_radius = config.get('base_radius', 20)
        self.difficulty = config.get('difficulty', 0.5)
        self.max_velocity = config.get('max_velocity', 5.0)
        self.tag_duration = config.get('tag_duration', 20)
        self.team_scores = [0, 0]  # Team 0 and Team 1
        self.win_score = config.get('win_score', 3)
        self.debug_level = config.get('debug_level', 1)  # Default to minimal debugging
        
        # Initialize game state
        self.agents: List[Agent] = []
        self.flags: List[Flag] = []
        self.team_bases: Dict[int, np.ndarray] = {}
        self.territories: Dict[int, List[np.ndarray]] = {}
        self.step_count = 0
        self.game_state = GameState.PLAYING
        
        # Set up the game
        self._setup_game()
        
        # Initialize rendering
        self.fig = None
        self.ax = None
        
        # Flag positions for respawn
        self.flag_positions = {flag.team: flag.position.copy() for flag in self.flags}
        
        # Spawn positions for agents
        self.spawn_positions = {0: self.team_bases[0].copy(), 1: self.team_bases[1].copy()}
        
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
        agent_id = 0
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
                    team=team,
                    id=agent_id
                ))
                agent_id += 1
                
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
        
    def step(self, action):
        """Take a step in the environment."""
        # Update agent state
        for i, agent in enumerate(self.agents):
            # Get appropriate action for this agent
            agent_action = action if i == 0 else self._get_opponent_action(agent)
            agent_action = np.clip(agent_action, -1, 1)  # Clip to valid range
            
            # Store the action for reference
            agent.last_action = agent_action
            
            # Only update position if not tagged
            if not agent.is_tagged:
                # Apply action to update position
                self._apply_action(i, agent_action)
            
        # Check if agents with flags have reached their base
        self._check_scoring()
            
        # Update tag timers for tagged agents
        for agent in self.agents:
            if agent.is_tagged:
                agent.tag_timer -= 1
                if agent.tag_timer <= 0:
                    agent.is_tagged = False
                    # Respawn at base when untagged
                    agent.position = self.team_bases[agent.team].copy() + np.random.uniform(-5, 5, 2)
                    agent.position = np.clip(agent.position, [0, 0], self.map_size)
                    
        # Update step count
        self.step_count += 1
        
        # Check if episode is done
        done = False
        winner = None
        
        # Game ends if max steps reached
        if self.step_count >= self.max_steps:
            done = True
            if self.team_scores[0] > self.team_scores[1]:
                self.game_state = GameState.WON
                winner = 0
            elif self.team_scores[1] > self.team_scores[0]:
                self.game_state = GameState.LOST
                winner = 1
            else:
                self.game_state = GameState.DRAW
        
        # Game ends if score threshold reached
        for team, score in enumerate(self.team_scores):
            if score >= self.win_score:
                done = True
                winner = team
                if team == 0:
                    self.game_state = GameState.WON
                else:
                    self.game_state = GameState.LOST
                
        # Calculate reward
        reward = self._calculate_reward(self.agents[0])
        
        # Only print reward details occasionally
        if self.debug_level >= 2 and np.random.random() < 0.01:  # Reduced from 0.05 to 0.01
            self._debug_reward(self.agents[0], reward)
        
        # Get observation
        observation = self._get_observation()
        
        # Get info
        info = self._get_info()
        
        return observation, reward, done, info
        
    def _apply_action(self, agent_idx: int, action: np.ndarray):
        """
        Apply action to update agent position and handle collisions
        
        Args:
            agent_idx: Index of the agent
            action: Action to apply (normalized direction vector)
        """
        agent = self.agents[agent_idx]
        
        # Normalize action if needed
        if np.linalg.norm(action) > 1.0:
            action = action / np.linalg.norm(action)
        
        # Set agent velocity
        agent.velocity = action * self.max_velocity
        
        # Calculate new position (slower if tagged)
        speed = self.max_velocity * (0.5 if agent.is_tagged else 1.0)
        new_position = agent.position + action * speed
        
        # Boundary check to keep agents inside the map
        new_position = np.clip(new_position, [0, 0], self.map_size)
        
        # Store previous position for collision resolution
        prev_position = agent.position.copy()
        agent.position = new_position
        
        # Handle collisions with other agents
        self._resolve_agent_collisions(agent_idx, prev_position)
        
        # Handle flag interactions (capture/pickup)
        self._handle_flag_interactions(agent)
        
        # Update tagging state
        self._handle_agent_tagging(agent)
        
    def _resolve_agent_collisions(self, agent_idx: int, prev_position: np.ndarray):
        """
        Resolve collisions between agents with improved physics
        
        Args:
            agent_idx: Index of the agent to check
            prev_position: Previous position of the agent before movement
        """
        agent = self.agents[agent_idx]
        
        # Early return if agent is tagged (no collision resolution needed)
        if agent.is_tagged:
            return
        
        # Define collision radius (can be agent size or a bit larger for smoother movement)
        collision_radius = 5.0
        
        # Check for collisions with other agents
        for other_idx, other_agent in enumerate(self.agents):
            if agent_idx == other_idx:  # Skip self
                continue
                
            # Calculate distance between agents
            distance = np.linalg.norm(agent.position - other_agent.position)
            
            # Check if collision occurred
            if distance < collision_radius * 2:  # Collision detected
                # Calculate collision response direction
                collision_dir = agent.position - other_agent.position
                
                # Avoid division by zero
                if np.linalg.norm(collision_dir) > 0:
                    collision_dir = collision_dir / np.linalg.norm(collision_dir)
                else:
                    # Agents at exact same position - push in random direction
                    angle = np.random.uniform(0, 2 * np.pi)
                    collision_dir = np.array([np.cos(angle), np.sin(angle)])
                
                # Calculate push strength based on overlap
                overlap = collision_radius * 2 - distance
                push_strength = max(0, overlap) * 0.5  # Scale factor to avoid excessive pushback
                
                # Move the current agent away from collision
                agent.position += collision_dir * push_strength
                
                # Ensure agent stays within map boundaries
                agent.position = np.clip(agent.position, [0, 0], self.map_size)
        
    def _handle_flag_interactions(self, agent):
        """
        Handle flag capture and drop logic
        
        Args:
            agent: The agent to check for flag interactions
        """
        # Skip if agent is tagged
        if agent.is_tagged:
            return
            
        # Check for flag pickup
        if not agent.has_flag:
            for flag in self.flags:
                if flag.team != agent.team and not flag.is_captured:
                    # Calculate distance to flag
                    distance = np.linalg.norm(agent.position - flag.position)
                    
                    # Check if agent can capture the flag
                    if distance < self.capture_radius:
                        flag.is_captured = True
                        flag.carrier_id = agent.id
                        agent.has_flag = True
                        if self.debug_level >= 1:
                            print(f"Flag captured by agent {agent.id} (team {agent.team})!")
                        break
    
    def _handle_agent_tagging(self, agent):
        """
        Handle agent tagging mechanics
        
        Args:
            agent: The agent to check for tagging interactions
        """
        # Skip if agent is tagged or has tag immunity
        if agent.is_tagged or agent.tag_timer < 0:
            return
            
        # Check for tagging opponents
        for other_agent in self.agents:
            # Skip self, tagged agents, and teammates
            if other_agent.id == agent.id or other_agent.is_tagged or other_agent.team == agent.team:
                continue
                
            # Calculate distance between agents
            distance = np.linalg.norm(agent.position - other_agent.position)
            
            # Check if within tagging range
            if distance < self.tag_radius:
                # Tag the other agent
                other_agent.is_tagged = True
                other_agent.tag_timer = self.tag_duration
                
                # If tagged agent had flag, drop it
                if other_agent.has_flag:
                    other_agent.has_flag = False
                    for flag in self.flags:
                        if flag.team != other_agent.team and flag.carrier_id == other_agent.id:
                            flag.is_captured = False
                            flag.carrier_id = None
                            flag.position = other_agent.position.copy()
                            
                # Update metrics
                if agent.team == 0:
                    # Increment tagging metric for player team
                    if hasattr(self, 'metrics'):
                        if 'tags_by_team_0' in self.metrics:
                            self.metrics['tags_by_team_0'] += 1
                
                if self.debug_level >= 2:
                    print(f"Agent {other_agent.id} (team {other_agent.team}) tagged by agent {agent.id}!")
                    
    def _check_scoring(self):
        """Check if agents with flags have reached their base to score."""
        for agent in self.agents:
            # Skip if agent doesn't have flag or is tagged
            if not agent.has_flag or agent.is_tagged:
                continue
                
            # Check if agent is at their base
            base_pos = self.team_bases[agent.team]
            dist = np.linalg.norm(agent.position - base_pos)
            
            if dist < self.base_radius:
                # Score a point
                self.team_scores[agent.team] += 1
                agent.has_flag = False
                
                # Reset the captured flag
                for flag in self.flags:
                    if flag.team != agent.team and flag.carrier_id == agent.id:
                        flag.is_captured = False
                        flag.carrier_id = None
                        flag.position = self.flag_positions[flag.team].copy()
                        
                if self.debug_level >= 1:
                    print(f"Team {agent.team} scored! New score: {self.team_scores}")
        
    def _calculate_reward(self, agent) -> float:
        """Calculate rewards for the current state."""
        reward = 0.0
        
        # Base reward for winning/losing
        if self.game_state == GameState.WON:
            reward += 10.0
        elif self.game_state == GameState.LOST:
            reward -= 10.0
            
        # Reward for scoring
        reward += self.team_scores[agent.team] * 5.0
        
        # Reward for capturing flag
        if agent.has_flag:
            reward += 3.0
            
        # Reward for tagging opponents
        reward += sum(1.0 for other in self.agents if other.team != agent.team and other.is_tagged)
        
        # Penalty for being tagged
        if agent.is_tagged:
            reward -= 2.0
        
        # Reward for approaching flag/base
        if not agent.has_flag:
            # Find closest enemy flag
            enemy_flags = [flag for flag in self.flags if flag.team != agent.team and not flag.is_captured]
            if enemy_flags:
                closest_flag = min(enemy_flags, key=lambda f: np.linalg.norm(agent.position - f.position))
                dist_to_flag = np.linalg.norm(agent.position - closest_flag.position)
                # Use map diagonal as normalization factor
                map_diagonal = np.linalg.norm(self.map_size)
                reward += max(0, 1.0 - dist_to_flag / map_diagonal) * 0.5
        else:
            # Reward for approaching own base with flag
            base_pos = self.team_bases[agent.team]
            dist_to_base = np.linalg.norm(agent.position - base_pos)
            # Use map diagonal as normalization factor
            map_diagonal = np.linalg.norm(self.map_size)
            reward += max(0, 1.0 - dist_to_base / map_diagonal) * 1.0
        
        # Small reward for being in opponent territory
        if self._is_in_territory(agent.position, 1 - agent.team):
            reward += 0.2
            
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
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            self.ax.set_xlim(0, self.map_size[0])
            self.ax.set_ylim(0, self.map_size[1])
            self.ax.set_aspect('equal')
            
        # Clear previous frame
        self.ax.clear()
        self.ax.set_xlim(0, self.map_size[0])
        self.ax.set_ylim(0, self.map_size[1])
        
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

    def _debug_reward(self, agent, total_reward):
        """Print detailed reward breakdown for debugging."""
        # Only print detailed reward at higher debug levels
        if self.debug_level < 2:
            return
            
        print("\nReward calculation debug:")
        
        # Base reward for winning/losing
        win_reward = 0
        if self.game_state == GameState.WON:
            win_reward = 10.0
            print(f"  Win reward: +10.0")
        elif self.game_state == GameState.LOST:
            win_reward = -10.0
            print(f"  Loss penalty: -10.0")
            
        # Reward for scoring
        score_reward = self.team_scores[0] * 5.0
        print(f"  Scoring reward: +{score_reward:.1f} ({self.team_scores[0]} points x 5.0)")
        
        # Reward for capturing flag
        flag_reward = 0
        if agent.has_flag:
            flag_reward = 3.0
            print(f"  Flag capture reward: +3.0")
            
        # Reward for tagging opponents
        tag_reward = sum(1.0 for other in self.agents if other.team != agent.team and other.is_tagged)
        print(f"  Tagging reward: +{tag_reward:.1f} ({sum(1 for other in self.agents if other.team != agent.team and other.is_tagged)} tags)")
        
        # Penalty for being tagged
        tag_penalty = -2.0 if agent.is_tagged else 0
        print(f"  Tag penalty: {tag_penalty:.1f}")
        
        # Use map diagonal as normalization factor
        map_diagonal = np.linalg.norm(self.map_size)
        
        # Reward for approaching flag/base
        approach_reward = 0
        if not agent.has_flag:
            # Find closest enemy flag
            enemy_flags = [flag for flag in self.flags if flag.team != agent.team and not flag.is_captured]
            if enemy_flags:
                closest_flag = min(enemy_flags, key=lambda f: np.linalg.norm(agent.position - f.position))
                dist_to_flag = np.linalg.norm(agent.position - closest_flag.position)
                # Use map diagonal as normalization factor
                approach_reward += max(0, 1.0 - dist_to_flag / map_diagonal) * 0.5
                print(f"  Approaching flag reward: +{max(0, 1.0 - dist_to_flag / map_diagonal) * 0.5:.2f} (distance: {dist_to_flag:.1f})")
        else:
            # Reward for approaching own base with flag
            base_pos = self.team_bases[agent.team]
            dist_to_base = np.linalg.norm(agent.position - base_pos)
            approach_reward += max(0, 1.0 - dist_to_base / map_diagonal) * 1.0
            print(f"  Approaching base with flag reward: +{max(0, 1.0 - dist_to_base / map_diagonal) * 1.0:.2f} (distance: {dist_to_base:.1f})")
            
        # Sum of rewards
        calculated_reward = win_reward + score_reward + flag_reward + tag_reward + approach_reward + tag_penalty
        print(f"  Calculated reward: {calculated_reward:.2f}")
        print(f"  Actual reward returned: {total_reward:.2f}")

    def _get_opponent_action(self, agent):
        """Generate action for opponent agent."""
        # Check if the agent velocity was already set by an external opponent strategy
        if agent.velocity is not None and np.any(agent.velocity != 0):
            # Return normalized action (velocity direction)
            norm = np.linalg.norm(agent.velocity)
            if norm > 0:
                return agent.velocity / (self.max_velocity or 1.0)
            return np.zeros(2)
            
        # Simple rule-based AI for opponents
        action = np.zeros(2)
        
        if agent.has_flag:
            # Return to base with flag
            base_pos = self.team_bases[agent.team]
            direction = base_pos - agent.position
        elif agent.is_tagged:
            # Can't move if tagged
            return np.zeros(2)
        else:
            # Find closest enemy flag that's not captured
            enemy_flags = [flag for flag in self.flags if flag.team != agent.team and not flag.is_captured]
            
            if enemy_flags:
                # Go for closest flag
                closest_flag = min(enemy_flags, key=lambda f: np.linalg.norm(agent.position - f.position))
                direction = closest_flag.position - agent.position
            else:
                # All flags captured, hunt enemy agents
                enemy_agents = [other for other in self.agents if other.team != agent.team and not other.is_tagged]
                if enemy_agents:
                    closest_enemy = min(enemy_agents, key=lambda a: np.linalg.norm(agent.position - a.position))
                    direction = closest_enemy.position - agent.position
                else:
                    # No valid targets, move randomly
                    direction = np.random.uniform(-1, 1, 2)
        
        # Normalize direction
        if np.linalg.norm(direction) > 0:
            action = direction / np.linalg.norm(direction)
        
        # Add some randomness
        action += np.random.normal(0, 0.1, 2)
        
        # Clip to valid range
        return np.clip(action, -1, 1) 