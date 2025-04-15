import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from typing import Dict, Any, List
import matplotlib.animation as animation

class HRLVisualizer:
    """Visualizer for HRL agent behavior and movement."""
    
    def __init__(self, env_bounds: List[float]):
        """
        Initialize the visualizer.
        
        Args:
            env_bounds: Environment bounds [width, height]
        """
        self.env_bounds = env_bounds
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_xlim(0, env_bounds[0])
        self.ax.set_ylim(0, env_bounds[1])
        self.ax.set_aspect('equal')
        
        # Initialize patches
        self.agent_patch = None
        self.opponent_patch = None
        self.team_flag_patch = None
        self.opponent_flag_patch = None
        self.patrol_points = []
        self.heatmap = np.zeros((int(env_bounds[0]), int(env_bounds[1])))
        
    def update_state(self, state: Dict[str, Any], option: str = None):
        """
        Update the visualization with current state.
        
        Args:
            state: Current environment state
            option: Current option being executed
        """
        # Clear previous patches
        self.ax.clear()
        self.ax.set_xlim(0, self.env_bounds[0])
        self.ax.set_ylim(0, self.env_bounds[1])
        
        # Update heatmap
        pos = state['agent_position']
        self.heatmap[int(pos[0]), int(pos[1])] += 1
        
        # Plot heatmap
        self.ax.imshow(self.heatmap.T, origin='lower', alpha=0.3,
                      extent=[0, self.env_bounds[0], 0, self.env_bounds[1]])
        
        # Plot agent
        agent_pos = state['agent_position']
        agent_heading = state['agent_heading']
        self.agent_patch = Circle(agent_pos, 2, color='blue')
        self.ax.add_patch(self.agent_patch)
        
        # Plot heading
        heading_length = 5
        heading_x = agent_pos[0] + heading_length * np.cos(np.radians(agent_heading))
        heading_y = agent_pos[1] + heading_length * np.sin(np.radians(agent_heading))
        self.ax.plot([agent_pos[0], heading_x], [agent_pos[1], heading_y], 'b-')
        
        # Plot opponent
        if 'opponent_position' in state:
            opponent_pos = state['opponent_position']
            opponent_heading = state['opponent_heading']
            self.opponent_patch = Circle(opponent_pos, 2, color='red')
            self.ax.add_patch(self.opponent_patch)
            
            # Plot opponent heading
            heading_x = opponent_pos[0] + heading_length * np.cos(np.radians(opponent_heading))
            heading_y = opponent_pos[1] + heading_length * np.sin(np.radians(opponent_heading))
            self.ax.plot([opponent_pos[0], heading_x], [opponent_pos[1], heading_y], 'r-')
        
        # Plot flags
        if 'team_flag_position' in state:
            team_flag_pos = state['team_flag_position']
            self.team_flag_patch = Rectangle(
                (team_flag_pos[0] - 1, team_flag_pos[1] - 1),
                2, 2, color='blue', alpha=0.5
            )
            self.ax.add_patch(self.team_flag_patch)
            
        if 'opponent_flag_position' in state:
            opponent_flag_pos = state['opponent_flag_position']
            self.opponent_flag_patch = Rectangle(
                (opponent_flag_pos[0] - 1, opponent_flag_pos[1] - 1),
                2, 2, color='red', alpha=0.5
            )
            self.ax.add_patch(self.opponent_flag_patch)
        
        # Plot patrol points if guard option is active
        if option == "guard_flag" and hasattr(self, 'patrol_points'):
            for point in self.patrol_points:
                self.ax.plot(point[0], point[1], 'gx')
        
        # Add title with current option
        if option:
            self.ax.set_title(f"Current Option: {option}")
        
        plt.draw()
        plt.pause(0.01)
        
    def set_patrol_points(self, points: List[np.ndarray]):
        """
        Set patrol points for visualization.
        
        Args:
            points: List of patrol point positions
        """
        self.patrol_points = points
        
    def save_animation(self, states: List[Dict[str, Any]], options: List[str], filename: str):
        """
        Save an animation of the agent's behavior.
        
        Args:
            states: List of states
            options: List of options executed
            filename: Output filename
        """
        def update(frame):
            self.update_state(states[frame], options[frame])
            return self.ax.patches
            
        anim = animation.FuncAnimation(
            self.fig, update, frames=len(states),
            interval=100, blit=True
        )
        anim.save(filename, writer='pillow', fps=10)
        
    def plot_trajectory(self, states: List[Dict[str, Any]]):
        """
        Plot the agent's trajectory.
        
        Args:
            states: List of states
        """
        positions = np.array([state['agent_position'] for state in states])
        self.ax.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.5)
        self.ax.plot(positions[0, 0], positions[0, 1], 'go', label='Start')
        self.ax.plot(positions[-1, 0], positions[-1, 1], 'ro', label='End')
        self.ax.legend()
        plt.show()
        
    def plot_heatmap(self):
        """Plot the agent's movement heatmap."""
        plt.figure(figsize=(10, 6))
        plt.imshow(self.heatmap.T, origin='lower', alpha=0.7,
                  extent=[0, self.env_bounds[0], 0, self.env_bounds[1]])
        plt.colorbar(label='Visit Count')
        plt.title('Agent Movement Heatmap')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.show() 