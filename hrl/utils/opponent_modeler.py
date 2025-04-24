import numpy as np
from typing import Dict, List, Any, Tuple
from enum import Enum, auto
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque
import math


class OpponentStrategy(Enum):
    """Possible opponent strategies."""
    UNKNOWN = auto()
    RANDOM = auto()
    AGGRESSIVE = auto()
    DEFENSIVE = auto()
    FLAG_FOCUSED = auto()
    COORDINATED = auto()


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
        self.history_length = config.get('history_length', 60)  # Number of steps to track
        
        # Track positions and actions of opponents
        self.position_history = [deque(maxlen=self.history_length) for _ in range(config.get('num_opponents', 10))]
        self.approach_counts = [0] * config.get('num_opponents', 10)
        self.flag_approach_counts = [0] * config.get('num_opponents', 10)
        self.defense_counts = [0] * config.get('num_opponents', 10)
        self.coordination_counts = [0] * config.get('num_opponents', 10)
        self.identified_strategies = [OpponentStrategy.UNKNOWN] * config.get('num_opponents', 10)
        self.strategy_confidence = [0.0] * config.get('num_opponents', 10)
        self.danger_zones = []
        self.update_counter = 0
        
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
            OpponentStrategy.AGGRESSIVE: "defensive",
            OpponentStrategy.DEFENSIVE: "flag_focused",
            OpponentStrategy.FLAG_FOCUSED: "aggressive",
            OpponentStrategy.COORDINATED: "split_and_distract"
        }
        
    def reset(self):
        """Reset all tracking stats for a new episode"""
        # Position history for each opponent
        for i in range(len(self.position_history)):
            self.position_history[i].clear()
        self.approach_counts = [0] * len(self.position_history)
        self.flag_approach_counts = [0] * len(self.position_history)
        self.defense_counts = [0] * len(self.position_history)
        self.coordination_counts = [0] * len(self.position_history)
        self.identified_strategies = [OpponentStrategy.UNKNOWN] * len(self.position_history)
        self.strategy_confidence = [0.0] * len(self.position_history)
        self.danger_zones = []
        self.update_counter = 0
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
        self.update_counter += 1
        
        # Extract relevant information from game state
        opponent_positions = []
        opponent_has_flags = []
        our_positions = []
        our_flag_position = None
        enemy_flag_position = None
        
        # Get positions of all agents
        for agent in state['agents']:
            if agent['team'] == 1:  # Enemy team
                opponent_positions.append(agent['position'])
                opponent_has_flags.append(agent['has_flag'])
            else:  # Our team
                our_positions.append(agent['position'])
                
        # Get flag positions
        for flag in state['flags']:
            if flag['team'] == 0:  # Our flag
                our_flag_position = flag['position']
            else:  # Enemy flag
                enemy_flag_position = flag['position']
                
        if not opponent_positions:
            return  # No opponents to track
            
        # Update position history
        for i, pos in enumerate(opponent_positions):
            if i < len(self.position_history):
                self.position_history[i].append(pos)
            
        # Only update strategy analysis every 10 steps
        if self.update_counter % 10 == 0:
            self._analyze_strategies(opponent_positions, opponent_has_flags, 
                                    our_positions, our_flag_position, enemy_flag_position)
            
        # Update danger zones
        self._update_danger_zones()
        
        # Identify strategies based on updated stats
        self._identify_strategies(state)
        
        # Update team-level strategy
        self._identify_team_strategy(state)
        
        # Debug output
        if self.debug_level >= 2 and state['step_count'] % 50 == 0:
            self._print_strategy_debug()
            
    def _analyze_strategies(self, opponent_positions, opponent_has_flags, 
                           our_positions, our_flag_position, enemy_flag_position):
        """Analyze opponent behaviors to identify strategies"""
        # Skip if we don't have all the necessary information
        if not opponent_positions or not our_positions or not our_flag_position or not enemy_flag_position:
            return
            
        # For each opponent
        for i, pos in enumerate(opponent_positions):
            if i >= len(self.position_history):
                break
                
            # Check if opponent is approaching our agents
            min_dist_to_our_agents = min(np.linalg.norm(np.array(pos) - np.array(our_pos)) 
                                        for our_pos in our_positions)
            if min_dist_to_our_agents < 5.0:  # Threshold for "approaching"
                self.approach_counts[i] += 1
                
            # Check if opponent is approaching our flag
            dist_to_our_flag = np.linalg.norm(np.array(pos) - np.array(our_flag_position))
            if dist_to_our_flag < 8.0:  # Threshold for "approaching flag"
                self.flag_approach_counts[i] += 1
                
            # Check if opponent is defending their flag
            dist_to_their_flag = np.linalg.norm(np.array(pos) - np.array(enemy_flag_position))
            if dist_to_their_flag < 10.0:  # Threshold for "defending"
                self.defense_counts[i] += 1
                
            # Check for coordination with other opponents
            for j, other_pos in enumerate(opponent_positions):
                if i != j and j < len(self.position_history):
                    dist_between = np.linalg.norm(np.array(pos) - np.array(other_pos))
                    if dist_between < 12.0:  # Threshold for "coordinating"
                        self.coordination_counts[i] += 1
                        
        # Identify strategies based on observed behaviors
        self._identify_strategies()
        
    def _identify_strategies(self, state: Dict[str, Any]):
        """
        Identify strategies for each opponent.
        
        Args:
            state: Current game state
        """
        for i in range(len(self.position_history)):
            # Calculate behavior percentages
            total_observations = max(1, self.update_counter // 10)
            aggression_pct = self.approach_counts[i] / total_observations
            flag_focus_pct = self.flag_approach_counts[i] / total_observations
            defense_pct = self.defense_counts[i] / total_observations
            coordination_pct = self.coordination_counts[i] / total_observations
            
            # Determine primary strategy
            strategy = OpponentStrategy.UNKNOWN
            confidence = 0.3  # Default confidence
            
            if coordination_pct > 0.5 and (aggression_pct > 0.4 or flag_focus_pct > 0.4):
                strategy = OpponentStrategy.COORDINATED
                confidence = coordination_pct * 0.7 + max(aggression_pct, flag_focus_pct) * 0.3
            elif aggression_pct > 0.6:
                strategy = OpponentStrategy.AGGRESSIVE
                confidence = aggression_pct
            elif flag_focus_pct > 0.6:
                strategy = OpponentStrategy.FLAG_FOCUSED
                confidence = flag_focus_pct
            elif defense_pct > 0.6:
                strategy = OpponentStrategy.DEFENSIVE
                confidence = defense_pct
            elif max(aggression_pct, flag_focus_pct, defense_pct, coordination_pct) < 0.3:
                strategy = OpponentStrategy.RANDOM
                confidence = 0.5
                
            # Update strategy with confidence
            self.identified_strategies[i] = strategy
            self.strategy_confidence[i] = min(1.0, confidence)
        
    def _identify_team_strategy(self, state: Dict[str, Any]):
        """
        Identify team-level strategy.
        
        Args:
            state: Current game state
        """
        # Get strategies for all opponents on team 1
        team_strategies = [strategy for i, strategy in enumerate(self.identified_strategies)
                         if next((a for a in state['agents'] if a['id'] == i), {'team': -1})['team'] == 1]
        
        if not team_strategies:
            return
            
        # Check for coordinated strategy - all agents have same strategy
        if len(set(team_strategies)) == 1 and team_strategies[0] != OpponentStrategy.UNKNOWN:
            self.team_strategy[1] = team_strategies[0]
            return
            
        # Otherwise use most common strategy
        from collections import Counter
        strategy_counter = Counter(team_strategies)
        most_common = strategy_counter.most_common(1)
        if most_common:
            strategy, count = most_common[0]
            if strategy != OpponentStrategy.UNKNOWN:
                self.team_strategy[1] = strategy
                
    def _update_danger_zones(self):
        """Update danger zones based on opponent movement patterns"""
        # Clear previous danger zones
        self.danger_zones = []
        
        # Create heatmap from opponent positions
        for i in range(len(self.position_history)):
            if len(self.position_history[i]) < 3:
                continue
                
            # Get recent positions with higher weight for more recent ones
            positions = list(self.position_history[i])
            weights = [0.5 + 0.5 * (idx / len(positions)) for idx in range(len(positions))]
            
            # Add positions to danger zones with their weights and the opponent's strategy
            for pos, weight in zip(positions, weights):
                strategy = self.identified_strategies[i]
                # Higher danger for aggressive opponents
                danger_factor = 1.5 if strategy == OpponentStrategy.AGGRESSIVE else 1.0
                self.danger_zones.append((pos, weight * danger_factor, strategy))
    
    def get_opponent_strategies(self) -> List[Tuple[OpponentStrategy, float]]:
        """Return the identified strategies with confidence levels"""
        return [(self.identified_strategies[i], self.strategy_confidence[i]) 
                for i in range(len(self.position_history))]
    
    def suggest_counter_strategy(self) -> str:
        """Suggest a counter strategy based on identified opponent strategies"""
        # Count strategy occurrences
        strategy_counts = defaultdict(int)
        for strategy in self.identified_strategies:
            strategy_counts[strategy] += 1
            
        # Determine dominant strategy
        dominant_strategy = max(strategy_counts.items(), key=lambda x: x[1])[0] 
        if dominant_strategy == OpponentStrategy.UNKNOWN:
            # Find next most dominant
            strategies = list(strategy_counts.items())
            strategies.sort(key=lambda x: x[1], reverse=True)
            if len(strategies) > 1:
                dominant_strategy = strategies[1][0]
            
        # Suggest counter strategy
        return self.counter_strategies.get(dominant_strategy, "balanced")
    
    def calculate_danger_score(self, position) -> float:
        """Calculate danger score for a given position"""
        if not self.danger_zones:
            return 0.0
            
        # Calculate danger based on proximity to danger zones
        danger_score = 0.0
        position = np.array(position)
        
        for zone_pos, weight, strategy in self.danger_zones:
            dist = np.linalg.norm(position - np.array(zone_pos))
            if dist < 15.0:  # Only consider nearby zones
                # Danger decreases with distance
                zone_danger = weight * (1.0 - min(1.0, dist / 15.0))
                danger_score += zone_danger
                
        # Normalize danger score
        return min(1.0, danger_score / 3.0)
    
    def suggest_safe_path(self, start_pos, goal_pos, map_size, step_size=5.0, samples=10) -> List[Tuple[float, float]]:
        """Suggest a safe path from start to goal avoiding danger zones"""
        if not self.danger_zones:
            # Direct path if no danger
            return [start_pos, goal_pos]
            
        start_pos = np.array(start_pos)
        goal_pos = np.array(goal_pos)
        direct_dist = np.linalg.norm(goal_pos - start_pos)
        
        # If very close to goal, go direct
        if direct_dist < 10.0:
            return [start_pos, goal_pos]
        
        # Try different paths and select safest
        best_path = [start_pos, goal_pos]
        best_safety = -1.0
        
        for _ in range(samples):
            # Generate a random midpoint
            angle = np.random.uniform(0, 2 * np.pi)
            offset_dist = np.random.uniform(0.3, 0.7) * direct_dist
            midpoint = start_pos + offset_dist * np.array([np.cos(angle), np.sin(angle)])
            
            # Ensure midpoint is within map bounds
            midpoint[0] = np.clip(midpoint[0], 0, map_size[0])
            midpoint[1] = np.clip(midpoint[1], 0, map_size[1])
            
            # Evaluate path safety
            path = [start_pos, midpoint, goal_pos]
            path_safety = self._evaluate_path_safety(path)
            
            if path_safety > best_safety:
                best_path = path
                best_safety = path_safety
                
        return best_path
    
    def _evaluate_path_safety(self, path) -> float:
        """Evaluate the safety of a path based on danger zones"""
        total_safety = 0.0
        
        # Check safety at multiple points along the path
        for i in range(len(path) - 1):
            start = np.array(path[i])
            end = np.array(path[i+1])
            dist = np.linalg.norm(end - start)
            
            # Sample points along this segment
            num_samples = max(2, int(dist / 5.0))
            for j in range(num_samples):
                t = j / (num_samples - 1)
                point = start + t * (end - start)
                danger = self.calculate_danger_score(point)
                segment_safety = 1.0 - danger
                total_safety += segment_safety
                
        # Average safety score
        return total_safety / (len(path) - 1)
        
    def _print_strategy_debug(self):
        """Print debug information about identified strategies."""
        if self.debug_level < 2:
            return
            
        print("\nOpponent Strategy Analysis:")
        print("-" * 40)
        
        # Individual agent strategies
        print("Individual Agents:")
        for i, strategy in enumerate(self.identified_strategies):
            stats = self.agent_stats.get(i, {})
            print(f"  Agent {i}: {strategy.name}")
            print(f"    Scores: AGG={stats.get('aggressive_score', 0):.1f}, DEF={stats.get('defensive_score', 0):.1f}, "
                 f"FLAG_FOCUSED={stats.get('flag_focused_score', 0):.1f}, COORD={stats.get('coordinated_score', 0):.1f}")
                 
        # Team strategy
        print("\nTeam Strategy:")
        print(f"  Team 1: {self.team_strategy[1].name}")
        print(f"  Recommended Counter: {self.suggest_counter_strategy()}")
        print("-" * 40) 