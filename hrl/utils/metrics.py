import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os

class MetricsTracker:
    """Tracks and logs various performance metrics during training and evaluation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the metrics tracker.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - log_dir: Directory to save logs (default: 'logs')
                - log_interval: Steps between logging (default: 100)
                - eval_interval: Steps between evaluations (default: 1000)
                - metrics_to_track: List of metrics to track (default: all)
                - save_replays: Whether to save game replays (default: False)
        """
        self.config = config or {}
        self.log_dir = self.config.get('log_dir', 'logs')
        self.log_interval = self.config.get('log_interval', 100)
        self.eval_interval = self.config.get('eval_interval', 1000)
        self.metrics_to_track = self.config.get('metrics_to_track', [
            'win_rate', 'score', 'flag_captures', 'tags', 'deaths',
            'option_usage', 'option_success', 'episode_length'
        ])
        self.save_replays = self.config.get('save_replays', False)
        
        # Initialize metrics storage
        self.metrics = {
            'episode': [],
            'step': [],
            'timestamp': [],
            'win_rate': [],
            'score': [],
            'flag_captures': [],
            'tags': [],
            'deaths': [],
            'option_usage': {},
            'option_success': {},
            'episode_length': [],
            'rewards': [],
            'q_values': [],
            'advantages': []
        }
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
    def update(self, step: int, episode: int, info: Dict[str, Any]):
        """
        Update metrics with new information.
        
        Args:
            step: Current training step
            episode: Current episode
            info: Dictionary containing metric information
        """
        # Record basic metrics
        self.metrics['episode'].append(episode)
        self.metrics['step'].append(step)
        self.metrics['timestamp'].append(datetime.now().isoformat())
        
        # Update game metrics
        if 'win' in info:
            self.metrics['win_rate'].append(float(info['win']))
        if 'score' in info:
            self.metrics['score'].append(info['score'])
        if 'flag_captures' in info:
            self.metrics['flag_captures'].append(info['flag_captures'])
        if 'tags' in info:
            self.metrics['tags'].append(info['tags'])
        if 'deaths' in info:
            self.metrics['deaths'].append(info['deaths'])
            
        # Update option metrics
        if 'option_usage' in info:
            for option, usage in info['option_usage'].items():
                if option not in self.metrics['option_usage']:
                    self.metrics['option_usage'][option] = []
                self.metrics['option_usage'][option].append(usage)
                
        if 'option_success' in info:
            for option, success in info['option_success'].items():
                if option not in self.metrics['option_success']:
                    self.metrics['option_success'][option] = []
                self.metrics['option_success'][option].append(success)
                
        # Update episode metrics
        if 'episode_length' in info:
            self.metrics['episode_length'].append(info['episode_length'])
        if 'rewards' in info:
            self.metrics['rewards'].append(info['rewards'])
        if 'q_values' in info:
            self.metrics['q_values'].append(info['q_values'])
        if 'advantages' in info:
            self.metrics['advantages'].append(info['advantages'])
            
        # Log metrics periodically
        if step % self.log_interval == 0:
            self._log_metrics(step)
            
        # Save replay if enabled
        if self.save_replays and 'replay' in info:
            self._save_replay(step, info['replay'])
            
    def _log_metrics(self, step: int):
        """Log current metrics to file."""
        # Calculate statistics
        stats = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'win_rate': np.mean(self.metrics['win_rate'][-100:]) if self.metrics['win_rate'] else 0.0,
            'avg_score': np.mean(self.metrics['score'][-100:]) if self.metrics['score'] else 0.0,
            'avg_flag_captures': np.mean(self.metrics['flag_captures'][-100:]) if self.metrics['flag_captures'] else 0.0,
            'avg_tags': np.mean(self.metrics['tags'][-100:]) if self.metrics['tags'] else 0.0,
            'avg_deaths': np.mean(self.metrics['deaths'][-100:]) if self.metrics['deaths'] else 0.0,
            'avg_episode_length': np.mean(self.metrics['episode_length'][-100:]) if self.metrics['episode_length'] else 0.0,
            'avg_reward': np.mean(self.metrics['rewards'][-100:]) if self.metrics['rewards'] else 0.0
        }
        
        # Add option statistics
        for option in self.metrics['option_usage']:
            stats[f'{option}_usage'] = np.mean(self.metrics['option_usage'][option][-100:])
            stats[f'{option}_success'] = np.mean(self.metrics['option_success'][option][-100:])
            
        # Save to log file
        log_file = os.path.join(self.log_dir, f'metrics_{step}.json')
        with open(log_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
    def _save_replay(self, step: int, replay_data: Dict[str, Any]):
        """Save game replay data."""
        replay_file = os.path.join(self.log_dir, f'replay_{step}.json')
        with open(replay_file, 'w') as f:
            json.dump(replay_data, f, indent=2)
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get all tracked metrics."""
        return self.metrics
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of all metrics."""
        stats = {}
        
        # Calculate basic statistics
        for metric in self.metrics:
            if isinstance(self.metrics[metric], list) and self.metrics[metric]:
                stats[f'{metric}_mean'] = np.mean(self.metrics[metric])
                stats[f'{metric}_std'] = np.std(self.metrics[metric])
                stats[f'{metric}_min'] = np.min(self.metrics[metric])
                stats[f'{metric}_max'] = np.max(self.metrics[metric])
                
        # Calculate option statistics
        for option in self.metrics['option_usage']:
            stats[f'{option}_usage_mean'] = np.mean(self.metrics['option_usage'][option])
            stats[f'{option}_success_mean'] = np.mean(self.metrics['option_success'][option])
            
        return stats
        
    def reset(self):
        """Reset all metrics."""
        for key in self.metrics:
            if isinstance(self.metrics[key], list):
                self.metrics[key] = []
            elif isinstance(self.metrics[key], dict):
                self.metrics[key] = {}
                
    def log_training_metric(self, metric_name: str, value: Any):
        """
        Log a training-related metric.
        
        Args:
            metric_name: Name of the metric
            value: Value to log
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
        
    def log_game_metric(self, metric_name: str, value: Any):
        """
        Log a game-related metric.
        
        Args:
            metric_name: Name of the metric
            value: Value to log
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
        
    def log_option_metric(self, metric_name: str, value: Any):
        """
        Log an option-related metric.
        
        Args:
            metric_name: Name of the metric
            value: Value to log
        """
        option_name = metric_name.split('_')[1] if '_' in metric_name else metric_name
        metric_type = 'option_usage' if 'usage' in metric_name else 'option_success'
        
        if option_name not in self.metrics[metric_type]:
            self.metrics[metric_type][option_name] = []
        self.metrics[metric_type][option_name].append(value)
        
    def save_metrics(self, filepath: str):
        """
        Save all metrics to a JSON file.
        
        Args:
            filepath: Path to save the metrics
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert metrics to serializable format
        serializable_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, list):
                serializable_metrics[key] = []
                for item in value:
                    if isinstance(item, np.ndarray):
                        serializable_metrics[key].append(item.tolist())
                    elif isinstance(item, (np.int32, np.int64, np.float32, np.float64)):
                        serializable_metrics[key].append(float(item))  # Convert to standard Python float
                    else:
                        serializable_metrics[key].append(item)
            elif isinstance(value, dict):
                serializable_metrics[key] = {}
                for k, v in value.items():
                    if not v:
                        serializable_metrics[key][k] = []
                        continue
                        
                    serializable_metrics[key][k] = []
                    for item in v:
                        if isinstance(item, np.ndarray):
                            serializable_metrics[key][k].append(item.tolist())
                        elif isinstance(item, (np.int32, np.int64, np.float32, np.float64)):
                            serializable_metrics[key][k].append(float(item))  # Convert to standard Python float
                        else:
                            serializable_metrics[key][k].append(item)
            else:
                # Handle individual numpy values
                if isinstance(value, (np.int32, np.int64, np.float32, np.float64)):
                    serializable_metrics[key] = float(value)
                else:
                    serializable_metrics[key] = value
                        
        # Save to file
        try:
            with open(filepath, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
        except TypeError as e:
            print(f"Error saving metrics: {e}")
            # Try a more aggressive conversion by converting the whole thing to str then back to basic types
            with open(filepath, 'w') as f:
                simple_metrics = {k: str(v) for k, v in serializable_metrics.items()}
                json.dump(simple_metrics, f, indent=2)
            
    def plot_metrics(self, metrics: List[str] = None):
        """
        Plot selected metrics over time.
        
        Args:
            metrics: List of metrics to plot (default: all)
        """
        try:
            import matplotlib.pyplot as plt
            
            if metrics is None:
                metrics = [m for m in self.metrics if isinstance(self.metrics[m], list)]
                
            fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 5*len(metrics)))
            if len(metrics) == 1:
                axes = [axes]
                
            for ax, metric in zip(axes, metrics):
                if metric in self.metrics and self.metrics[metric]:
                    ax.plot(self.metrics['step'], self.metrics[metric])
                    ax.set_title(metric)
                    ax.set_xlabel('Step')
                    ax.set_ylabel('Value')
                    
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, 'metrics_plot.png'))
            plt.close()
            
        except ImportError:
            print("Matplotlib not installed. Skipping plotting.")

# Create an alias for backward compatibility
Metrics = MetricsTracker 