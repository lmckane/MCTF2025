import numpy as np
from typing import Dict, List, Any, Tuple
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F


class OpponentStrategy(Enum):
    """Possible opponent strategies."""
    UNKNOWN = 0
    AGGRESSIVE = 1  # Focuses on tagging
    DEFENSIVE = 2   # Focuses on defending their flag
    RUSHING = 3     # Rapidly goes for flag
    FLANKING = 4    # Tries to approach from sides
    COORDINATED = 5 # Shows team coordination


class OpponentModeler:
    """
    Models opponent strategies and suggests counter-tactics.
    
    This class:
    1. Tracks opponent movements and actions
    2. Identifies patterns in behavior
    3. Classifies opponent strategies
    4. Suggests appropriate counter-strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the opponent modeler.
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config
        self.debug_level = config.get('debug_level', 1)
        self.history_length = config.get('history_length', 20)  # Number of steps to track
        
        # Track positions and actions of opponents
        self.opponent_history = {}  # {agent_id: [positions_history]}
        self.identified_strategies = {}  # {agent_id: OpponentStrategy}
        
        # Track team-level strategy
        self.team_strategy = {0: OpponentStrategy.UNKNOWN, 1: OpponentStrategy.UNKNOWN}
        
        # Statistics for strategy identification
        self.agent_stats = {}  # {agent_id: {'aggressive_score': 0, 'defensive_score': 0, ...}}
        
        # Initialize feature extractor for positions
        self.feature_size = 8
        self.feature_extractor = nn.Sequential(
            nn.Linear(4, 16),  # position and velocity
            nn.ReLU(),
            nn.Linear(16, self.feature_size)
        )
        
        # Record of flag captures and tags
        self.flag_capture_history = []
        self.tag_history = []
        
        # Counter-strategy suggestions
        self.counter_strategies = {
            OpponentStrategy.AGGRESSIVE: "evasive",  # Evade aggressive opponents
            OpponentStrategy.DEFENSIVE: "flanking",   # Flank defensive opponents
            OpponentStrategy.RUSHING: "intercept",   # Intercept rushing opponents
            OpponentStrategy.FLANKING: "defensive",  # Be defensive against flanking
            OpponentStrategy.COORDINATED: "disrupt"  # Disrupt coordinated teams
        }
        
    def reset(self):
        """Reset the modeler state for a new episode."""
        self.opponent_history = {}
        self.identified_strategies = {}
        self.team_strategy = {0: OpponentStrategy.UNKNOWN, 1: OpponentStrategy.UNKNOWN}
        self.agent_stats = {}
        self.flag_capture_history = []
        self.tag_history = []
        
    def update(self, state: Dict[str, Any]):
        """
        Update the opponent model based on new state.
        
        Args:
            state: Current game state
        """
        # Extract opponents (agents not on team 0)
        opponents = [agent for agent in state['agents'] if agent['team'] != 0]
        
        # Update position history for each opponent
        for opponent in opponents:
            agent_id = opponent['id']
            position = np.array(opponent['position'])
            velocity = np.array(opponent['velocity'])
            is_tagged = opponent['is_tagged']
            has_flag = opponent['has_flag']
            
            # Initialize history and stats if new opponent
            if agent_id not in self.opponent_history:
                self.opponent_history[agent_id] = []
                self.agent_stats[agent_id] = {
                    'aggressive_score': 0.0,
                    'defensive_score': 0.0,
                    'rushing_score': 0.0,
                    'flanking_score': 0.0,
                    'coordinated_score': 0.0,
                    'tags_initiated': 0,
                    'been_tagged': 0,
                    'flag_captures': 0,
                    'time_in_territory': 0,
                    'territory_violations': 0
                }
            
            # Record position and state
            self.opponent_history[agent_id].append({
                'position': position,
                'velocity': velocity,
                'is_tagged': is_tagged,
                'has_flag': has_flag,
                'step': state['step_count']
            })
            
            # Limit history length
            if len(self.opponent_history[agent_id]) > self.history_length:
                self.opponent_history[agent_id].pop(0)
            
            # Update stats based on current state
            self._update_agent_stats(agent_id, state)
        
        # Identify strategies based on updated stats
        self._identify_strategies(state)
        
        # Update team-level strategy
        self._identify_team_strategy(state)
        
        # Debug output
        if self.debug_level >= 2 and state['step_count'] % 50 == 0:
            self._print_strategy_debug()
            
    def _update_agent_stats(self, agent_id: int, state: Dict[str, Any]):
        """
        Update statistics for an opponent agent.
        
        Args:
            agent_id: ID of the agent
            state: Current game state
        """
        # Get opponent info
        opponent = next((a for a in state['agents'] if a['id'] == agent_id), None)
        if not opponent:
            return
            
        # Extract key information
        position = np.array(opponent['position'])
        team = opponent['team']
        
        # Get opponents' flag (our flag)
        our_flag = next((f for f in state['flags'] if f['team'] == 0), None)
        if our_flag:
            our_flag_pos = np.array(our_flag['position'])
            
            # Check if opponent is near our flag
            dist_to_our_flag = np.linalg.norm(position - our_flag_pos)
            if dist_to_our_flag < 20:  # Close to our flag
                self.agent_stats[agent_id]['rushing_score'] += 0.05
            
        # Get their flag
        their_flag = next((f for f in state['flags'] if f['team'] == team), None)
        if their_flag:
            their_flag_pos = np.array(their_flag['position'])
            
            # Check if opponent is defending their flag
            dist_to_their_flag = np.linalg.norm(position - their_flag_pos)
            if dist_to_their_flag < 20:  # Close to their flag
                self.agent_stats[agent_id]['defensive_score'] += 0.05
        
        # Check if in the middle of the map (flanking behavior)
        map_size = np.array(state.get('map_size', [100, 100]))
        map_center = map_size / 2
        dist_to_center = np.linalg.norm(position - map_center)
        if dist_to_center < 20:  # Near center - possible flanking behavior
            self.agent_stats[agent_id]['flanking_score'] += 0.02
        
        # Territory analysis
        if self._is_in_territory(position, 0, state):  # In our territory
            self.agent_stats[agent_id]['territory_violations'] += 1
            
            # Aggressive behavior in our territory
            self.agent_stats[agent_id]['aggressive_score'] += 0.03
            
        # Check for coordination with teammates
        teammate_positions = [np.array(a['position']) for a in state['agents'] 
                             if a['team'] == team and a['id'] != agent_id]
        if teammate_positions:
            avg_distance = np.mean([np.linalg.norm(position - tp) for tp in teammate_positions])
            if avg_distance < 30:  # Close to teammates - possible coordination
                self.agent_stats[agent_id]['coordinated_score'] += 0.04
                
    def _identify_strategies(self, state: Dict[str, Any]):
        """
        Identify strategies for each opponent.
        
        Args:
            state: Current game state
        """
        for agent_id, stats in self.agent_stats.items():
            # Simple strategy identification using the highest score
            strategy_scores = {
                OpponentStrategy.AGGRESSIVE: stats['aggressive_score'],
                OpponentStrategy.DEFENSIVE: stats['defensive_score'],
                OpponentStrategy.RUSHING: stats['rushing_score'],
                OpponentStrategy.FLANKING: stats['flanking_score'],
                OpponentStrategy.COORDINATED: stats['coordinated_score']
            }
            
            # Find strategy with highest score
            max_strategy = max(strategy_scores.items(), key=lambda x: x[1])
            
            # Only identify if score exceeds threshold
            if max_strategy[1] > 1.0:  # Sufficient evidence for strategy
                self.identified_strategies[agent_id] = max_strategy[0]
            else:
                self.identified_strategies[agent_id] = OpponentStrategy.UNKNOWN
                
    def _identify_team_strategy(self, state: Dict[str, Any]):
        """
        Identify team-level strategy.
        
        Args:
            state: Current game state
        """
        # Get strategies for all opponents on team 1
        team_strategies = [strategy for agent_id, strategy in self.identified_strategies.items()
                         if next((a for a in state['agents'] if a['id'] == agent_id), {'team': -1})['team'] == 1]
        
        if not team_strategies:
            return
            
        # Check for coordinated strategy - all agents have same strategy
        if len(set(team_strategies)) == 1 and team_strategies[0] != OpponentStrategy.UNKNOWN:
            self.team_strategy[1] = OpponentStrategy.COORDINATED
            return
            
        # Otherwise use most common strategy
        from collections import Counter
        strategy_counter = Counter(team_strategies)
        most_common = strategy_counter.most_common(1)
        if most_common:
            strategy, count = most_common[0]
            if strategy != OpponentStrategy.UNKNOWN:
                self.team_strategy[1] = strategy
                
    def _is_in_territory(self, position: np.ndarray, team: int, state: Dict[str, Any]) -> bool:
        """
        Check if a position is in a team's territory.
        
        Args:
            position: Position to check
            team: Team whose territory to check
            state: Game state
            
        Returns:
            True if in territory, False otherwise
        """
        territories = state['territories']
        territory = territories[team]
        
        # Simple polygon check using ray casting algorithm
        x, y = position
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
                
    def get_counter_strategy(self, agent_id: int = None) -> str:
        """
        Get suggested counter-strategy for an opponent.
        
        Args:
            agent_id: ID of the opponent (None for team strategy)
            
        Returns:
            Suggested counter-strategy name
        """
        if agent_id is not None and agent_id in self.identified_strategies:
            strategy = self.identified_strategies[agent_id]
        else:
            # Use team strategy if agent not specified or not identified
            strategy = self.team_strategy[1]
            
        # Return counter-strategy suggestion
        return self.counter_strategies.get(strategy, "balanced")
        
    def get_danger_score(self, position: np.ndarray, state: Dict[str, Any]) -> float:
        """
        Calculate danger score for a position based on opponent models.
        
        Args:
            position: Position to evaluate
            state: Current game state
            
        Returns:
            Danger score (0.0-1.0, higher = more dangerous)
        """
        opponents = [agent for agent in state['agents'] if agent['team'] != 0]
        if not opponents:
            return 0.0
            
        danger_score = 0.0
        
        for opponent in opponents:
            agent_id = opponent['id']
            op_position = np.array(opponent['position'])
            distance = np.linalg.norm(position - op_position)
            
            # Base danger from proximity
            proximity_danger = max(0.0, 1.0 - distance / 50.0)
            
            # Adjust danger based on identified strategy
            strategy_multiplier = 1.0
            if agent_id in self.identified_strategies:
                strategy = self.identified_strategies[agent_id]
                if strategy == OpponentStrategy.AGGRESSIVE:
                    strategy_multiplier = 1.5  # Aggressive opponents are more dangerous
                elif strategy == OpponentStrategy.DEFENSIVE:
                    strategy_multiplier = 0.7  # Defensive opponents less likely to chase
                    
            # Add to total danger score (weighted by proximity)
            danger_score = max(danger_score, proximity_danger * strategy_multiplier)
            
        return min(1.0, danger_score)  # Cap at 1.0
        
    def suggest_path(self, start: np.ndarray, goal: np.ndarray, state: Dict[str, Any]) -> List[np.ndarray]:
        """
        Suggest a safe path from start to goal avoiding dangers.
        
        Args:
            start: Starting position
            goal: Goal position
            state: Current game state
            
        Returns:
            List of waypoints for a safer path
        """
        # Direct path
        direct_vector = goal - start
        distance = np.linalg.norm(direct_vector)
        
        if distance < 1e-6:
            return [goal]  # Already at goal
            
        normalized_direct = direct_vector / distance
        
        # Check danger along direct path
        path_danger = 0.0
        num_samples = 5
        for i in range(1, num_samples + 1):
            sample_point = start + direct_vector * (i / (num_samples + 1))
            path_danger = max(path_danger, self.get_danger_score(sample_point, state))
            
        # If path is safe enough, just return direct path
        if path_danger < 0.3:
            return [goal]
            
        # Calculate perpendicular vector for flanking
        perp_vector = np.array([-direct_vector[1], direct_vector[0]])
        perp_vector = perp_vector / np.linalg.norm(perp_vector)
        
        # Check danger along left and right flanking paths
        left_flank = start + perp_vector * (distance * 0.3)  # 30% of distance perpendicular
        right_flank = start - perp_vector * (distance * 0.3)
        
        left_danger = self.get_danger_score(left_flank, state)
        right_danger = self.get_danger_score(right_flank, state)
        
        # Choose safer path
        if left_danger < right_danger and left_danger < path_danger:
            waypoint = left_flank
        elif right_danger < left_danger and right_danger < path_danger:
            waypoint = right_flank
        else:
            # If both flanks are dangerous, try farther flanks
            far_left = start + perp_vector * (distance * 0.6)
            far_right = start - perp_vector * (distance * 0.6)
            
            far_left_danger = self.get_danger_score(far_left, state)
            far_right_danger = self.get_danger_score(far_right, state)
            
            if far_left_danger < far_right_danger:
                waypoint = far_left
            else:
                waypoint = far_right
                
        # Check if flanking point is safe, otherwise try center-point between start/goal
        if self.get_danger_score(waypoint, state) > 0.5:
            midpoint = (start + goal) / 2
            waypoint = midpoint
        
        # Map boundaries
        map_size = np.array(state.get('map_size', [100, 100]))
        waypoint = np.clip(waypoint, [0, 0], map_size)
        
        # Return waypoint and goal
        return [waypoint, goal]
            
    def _print_strategy_debug(self):
        """Print debug information about identified strategies."""
        if self.debug_level < 2:
            return
            
        print("\nOpponent Strategy Analysis:")
        print("-" * 40)
        
        # Individual agent strategies
        print("Individual Agents:")
        for agent_id, strategy in self.identified_strategies.items():
            stats = self.agent_stats[agent_id]
            print(f"  Agent {agent_id}: {strategy.name}")
            print(f"    Scores: AGG={stats['aggressive_score']:.1f}, DEF={stats['defensive_score']:.1f}, "
                 f"RUSH={stats['rushing_score']:.1f}, FLANK={stats['flanking_score']:.1f}, "
                 f"COORD={stats['coordinated_score']:.1f}")
                 
        # Team strategy
        print("\nTeam Strategy:")
        print(f"  Team 1: {self.team_strategy[1].name}")
        print(f"  Recommended Counter: {self.get_counter_strategy()}")
        print("-" * 40) 