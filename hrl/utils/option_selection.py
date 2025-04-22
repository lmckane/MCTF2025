from typing import Dict, Any, List
import numpy as np
from hrl.utils.option_evaluation import OptionEvaluator
from hrl.utils.metrics_tracker import MetricsTracker

class OptionSelector:
    """Selects the most appropriate option for the current state."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.evaluator = OptionEvaluator(config)
        self.exploration_rate = config.get("exploration_rate", 0.1)
        
    def select_option(self, state: Dict[str, Any], 
                     available_options: List[str]) -> str:
        """
        Select the best option for the current state.
        
        Args:
            state: Current environment state
            available_options: List of available option names
            
        Returns:
            str: Selected option name
        """
        if not available_options:
            return None
            
        # Exploration: randomly select an option
        if np.random.random() < self.exploration_rate:
            return np.random.choice(available_options)
            
        # Exploitation: select best option
        option_scores = {}
        for option_name in available_options:
            score = self._get_option_score(state, option_name)
            option_scores[option_name] = score
            
        return max(option_scores.items(), key=lambda x: x[1])[0]
        
    def _get_option_score(self, state: Dict[str, Any], option_name: str) -> float:
        """Get score for option in current state."""
        # Get base score from evaluator
        base_score = self.evaluator.evaluate_option(option_name, [state])
        
        # Adjust score based on state features
        adjustments = self._get_state_adjustments(state, option_name)
        
        # Combine scores
        final_score = base_score * np.prod(adjustments)
        return max(0.0, min(1.0, final_score))
        
    def _get_state_adjustments(self, state: Dict[str, Any], option_name: str) -> List[float]:
        """Get adjustments to option score based on state features."""
        adjustments = []
        
        # Distance to relevant objects
        if option_name == "capture":
            # Prefer capture when close to opponent flag
            flag_dist = self._get_distance(state["agent_position"], 
                                         state["opponent_flag_position"])
            adjustments.append(1.0 / (1.0 + flag_dist))
            
        elif option_name == "defend":
            # Prefer defend when opponents are near own flag
            flag_pos = state["team_flag_position"]
            min_opp_dist = min(self._get_distance(flag_pos, opp_pos)
                             for opp_pos in state["opponent_positions"])
            adjustments.append(1.0 / (1.0 + min_opp_dist))
            
        elif option_name == "patrol":
            # Prefer patrol when in safe areas
            min_opp_dist = min(self._get_distance(state["agent_position"], opp_pos)
                             for opp_pos in state["opponent_positions"])
            adjustments.append(min_opp_dist / 10.0)
            
        return adjustments
        
    def _get_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate distance between two positions."""
        return np.linalg.norm(pos1 - pos2)
        
    def update_exploration_rate(self, episode: int, total_episodes: int):
        """Update exploration rate based on training progress."""
        # Linearly decrease exploration rate
        self.exploration_rate = max(0.01, 1.0 - (episode / total_episodes))

# Initialize metrics tracker
metrics = MetricsTracker(config={
    'log_dir': 'training_logs',
    'log_interval': 100,
    'save_replays': True
})

# In your training loop
for episode in range(num_episodes):
    # ... training code ...
    
    # Update metrics
    metrics.update(
        step=current_step,
        episode=episode,
        info={
            'win': episode_won,
            'score': episode_score,
            'flag_captures': num_captures,
            'tags': num_tags,
            'deaths': num_deaths,
            'option_usage': option_usage_stats,
            'option_success': option_success_stats,
            'episode_length': episode_length,
            'rewards': episode_rewards,
            'q_values': episode_q_values,
            'advantages': episode_advantages,
            'replay': episode_replay_data  # Optional
        }
    )
    
    # Plot metrics periodically
    if episode % 100 == 0:
        metrics.plot_metrics(['win_rate', 'score', 'flag_captures']) 