from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from datetime import datetime

class OptionVisualizer:
    """Provides visualization capabilities for option execution and performance."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fig_size = config.get("fig_size", (10, 8))
        self.color_palette = config.get("color_palette", "viridis")
        self.style = config.get("style", "seaborn")
        
        # Set style
        plt.style.use(self.style)
        sns.set_palette(self.color_palette)
        
    def plot_execution_timeline(self, option_name: str,
                              execution_logs: List[Dict[str, Any]]):
        """
        Plot execution timeline for an option.
        
        Args:
            option_name: Name of the option
            execution_logs: List of execution logs
        """
        if not execution_logs:
            return
            
        # Extract timestamps and states
        timestamps = [
            datetime.fromisoformat(log["timestamp"])
            for log in execution_logs
        ]
        states = [log["state"] for log in execution_logs]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Plot state values over time
        for key in states[0].keys():
            values = [state[key] for state in states]
            ax.plot(timestamps, values, label=key)
            
        # Customize plot
        ax.set_title(f"Execution Timeline - {option_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("State Values")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
        
    def plot_performance_metrics(self, option_name: str,
                               performance_logs: List[Dict[str, Any]]):
        """
        Plot performance metrics for an option.
        
        Args:
            option_name: Name of the option
            performance_logs: List of performance logs
        """
        if not performance_logs:
            return
            
        # Extract timestamps and metrics
        timestamps = [
            datetime.fromisoformat(log["timestamp"])
            for log in performance_logs
        ]
        metrics = [log["metrics"] for log in performance_logs]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Plot metrics over time
        for key in metrics[0].keys():
            values = [metric[key] for metric in metrics]
            ax.plot(timestamps, values, label=key)
            
        # Customize plot
        ax.set_title(f"Performance Metrics - {option_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Metric Values")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
        
    def plot_error_distribution(self, option_name: str,
                              error_logs: List[Dict[str, Any]]):
        """
        Plot error distribution for an option.
        
        Args:
            option_name: Name of the option
            error_logs: List of error logs
        """
        if not error_logs:
            return
            
        # Extract error types and counts
        error_types = defaultdict(int)
        for log in error_logs:
            error_type = log["error"].get("type", "unknown")
            error_types[error_type] += 1
            
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Plot error distribution
        error_types = dict(sorted(error_types.items(), key=lambda x: x[1], reverse=True))
        ax.bar(error_types.keys(), error_types.values())
        
        # Customize plot
        ax.set_title(f"Error Distribution - {option_name}")
        ax.set_xlabel("Error Type")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
        
    def plot_state_distribution(self, option_name: str,
                              execution_logs: List[Dict[str, Any]]):
        """
        Plot state distribution for an option.
        
        Args:
            option_name: Name of the option
            execution_logs: List of execution logs
        """
        if not execution_logs:
            return
            
        # Extract state values
        states = [log["state"] for log in execution_logs]
        
        # Create figure
        fig, axes = plt.subplots(
            nrows=len(states[0]),
            figsize=(self.fig_size[0], self.fig_size[1] * len(states[0])),
            squeeze=False
        )
        
        # Plot distribution for each state variable
        for i, key in enumerate(states[0].keys()):
            values = [state[key] for state in states]
            sns.histplot(values, ax=axes[i, 0], kde=True)
            axes[i, 0].set_title(f"Distribution of {key}")
            axes[i, 0].set_xlabel("Value")
            axes[i, 0].set_ylabel("Count")
            
        plt.tight_layout()
        
        return fig
        
    def plot_action_distribution(self, option_name: str,
                               execution_logs: List[Dict[str, Any]]):
        """
        Plot action distribution for an option.
        
        Args:
            option_name: Name of the option
            execution_logs: List of execution logs
        """
        if not execution_logs:
            return
            
        # Extract actions
        actions = [log["action"] for log in execution_logs]
        
        # Create figure
        fig, axes = plt.subplots(
            nrows=len(actions[0]),
            figsize=(self.fig_size[0], self.fig_size[1] * len(actions[0])),
            squeeze=False
        )
        
        # Plot distribution for each action variable
        for i, key in enumerate(actions[0].keys()):
            values = [action[key] for action in actions]
            sns.histplot(values, ax=axes[i, 0], kde=True)
            axes[i, 0].set_title(f"Distribution of {key}")
            axes[i, 0].set_xlabel("Value")
            axes[i, 0].set_ylabel("Count")
            
        plt.tight_layout()
        
        return fig
        
    def create_execution_animation(self, option_name: str,
                                 execution_logs: List[Dict[str, Any]],
                                 interval: int = 200):
        """
        Create animation of option execution.
        
        Args:
            option_name: Name of the option
            execution_logs: List of execution logs
            interval: Animation interval in milliseconds
            
        Returns:
            FuncAnimation: Animation object
        """
        if not execution_logs:
            return
            
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Initialize plot
        def init():
            ax.clear()
            ax.set_title(f"Execution Animation - {option_name}")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("State Values")
            return []
            
        # Update plot for each frame
        def update(frame):
            ax.clear()
            
            # Plot state values up to current frame
            for key in execution_logs[0]["state"].keys():
                values = [
                    log["state"][key]
                    for log in execution_logs[:frame+1]
                ]
                ax.plot(range(len(values)), values, label=key)
                
            ax.set_title(f"Execution Animation - {option_name}")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("State Values")
            ax.legend()
            
            return []
            
        # Create animation
        animation = FuncAnimation(
            fig, update, frames=len(execution_logs),
            init_func=init, interval=interval, blit=True
        )
        
        return animation
        
    def save_plot(self, fig, file_path: str):
        """
        Save plot to file.
        
        Args:
            fig: Figure object
            file_path: Path to save plot
        """
        fig.savefig(file_path)
        plt.close(fig)
        
    def save_animation(self, animation, file_path: str):
        """
        Save animation to file.
        
        Args:
            animation: Animation object
            file_path: Path to save animation
        """
        animation.save(file_path)
        plt.close(animation._fig) 