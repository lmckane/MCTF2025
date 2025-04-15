from typing import Dict, Any, List, Tuple
import numpy as np

class StateProcessor:
    """Processes and normalizes state information for the HRL system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state_bounds = config.get("state_bounds", {
            "position": [-100, 100],
            "velocity": [-10, 10],
            "heading": [-np.pi, np.pi]
        })
        self.normalize = config.get("normalize", True)
        
    def process_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and normalize the raw state.
        
        Args:
            state: Raw state dictionary
            
        Returns:
            Dict[str, Any]: Processed state
        """
        processed = {}
        
        # Process agent state
        if "agent_position" in state:
            processed["agent_position"] = self._normalize_position(state["agent_position"])
        if "agent_velocity" in state:
            processed["agent_velocity"] = self._normalize_velocity(state["agent_velocity"])
        if "agent_heading" in state:
            processed["agent_heading"] = self._normalize_heading(state["agent_heading"])
            
        # Process opponent states
        if "opponent_positions" in state:
            processed["opponent_positions"] = [
                self._normalize_position(pos) for pos in state["opponent_positions"]
            ]
            
        # Process flag states
        if "flag_position" in state:
            processed["flag_position"] = self._normalize_position(state["flag_position"])
        if "has_flag" in state:
            processed["has_flag"] = state["has_flag"]
            
        # Process game state
        if "game_time" in state:
            processed["game_time"] = self._normalize_time(state["game_time"])
        if "score" in state:
            processed["score"] = self._normalize_score(state["score"])
            
        # Add derived features
        processed.update(self._compute_derived_features(processed))
        
        return processed
        
    def _normalize_position(self, pos: np.ndarray) -> np.ndarray:
        """Normalize position coordinates."""
        if not self.normalize:
            return pos
            
        bounds = self.state_bounds["position"]
        return (pos - bounds[0]) / (bounds[1] - bounds[0])
        
    def _normalize_velocity(self, vel: np.ndarray) -> np.ndarray:
        """Normalize velocity components."""
        if not self.normalize:
            return vel
            
        bounds = self.state_bounds["velocity"]
        return (vel - bounds[0]) / (bounds[1] - bounds[0])
        
    def _normalize_heading(self, heading: float) -> float:
        """Normalize heading angle."""
        if not self.normalize:
            return heading
            
        bounds = self.state_bounds["heading"]
        return (heading - bounds[0]) / (bounds[1] - bounds[0])
        
    def _normalize_time(self, time: float) -> float:
        """Normalize game time."""
        max_time = self.config.get("max_game_time", 300)
        return time / max_time
        
    def _normalize_score(self, score: int) -> float:
        """Normalize game score."""
        max_score = self.config.get("max_score", 10)
        return score / max_score
        
    def _compute_derived_features(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Compute additional features from the state."""
        features = {}
        
        # Compute distances
        if "agent_position" in state and "flag_position" in state:
            features["flag_distance"] = self._get_distance(
                state["agent_position"], state["flag_position"]
            )
            
        if "agent_position" in state and "opponent_positions" in state:
            features["opponent_distances"] = [
                self._get_distance(state["agent_position"], opp_pos)
                for opp_pos in state["opponent_positions"]
            ]
            features["nearest_opponent_distance"] = min(features["opponent_distances"])
            
        # Compute relative positions
        if "agent_position" in state and "flag_position" in state:
            features["flag_relative_position"] = (
                state["flag_position"] - state["agent_position"]
            )
            
        # Compute safety metrics
        if "nearest_opponent_distance" in features:
            features["safety_score"] = 1.0 / (1.0 + features["nearest_opponent_distance"])
            
        return features
        
    def _get_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate distance between two positions."""
        return np.linalg.norm(pos1 - pos2) 