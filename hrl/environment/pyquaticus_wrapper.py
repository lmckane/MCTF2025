"""
Wrapper for PyquaticusEnv to make it compatible with the training infrastructure.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import copy

from pyquaticus import pyquaticus_v0
from pyquaticus.envs.pyquaticus import Team
import pyquaticus.config
import pyquaticus.utils.rewards as rewards

class PyquaticusWrapper:
    """
    Wrapper for PyquaticusEnv to make it interface like the custom GameEnvironment.
    This allows training code to use the official Pyquaticus environment.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Pyquaticus environment wrapper.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.debug_level = config.get('debug_level', 1)
        
        # Map configuration parameters to Pyquaticus config
        pyq_config = copy.deepcopy(pyquaticus.config.config_dict_std)
        
        # Apply configuration overrides
        pyq_config['env_bounds'] = config.get('map_size', [100, 100])
        pyq_config['max_time'] = config.get('max_steps', 1000)
        pyq_config['max_score'] = config.get('win_score', 3)
        
        # IMPORTANT: In Pyquaticus, catch_radius MUST be greater than flag_keepout
        # Default values in Pyquaticus std config are:
        # catch_radius = 10.0, flag_keepout = 3.0
        # So we'll maintain that ratio
        pyq_config['catch_radius'] = 10.0  # Use Pyquaticus default
        pyq_config['flag_keepout'] = 3.0   # Use Pyquaticus default
        
        pyq_config['flag_radius'] = 2.0  # Use Pyquaticus default
        pyq_config['tagging_cooldown'] = config.get('tag_cooldown', 60)
        pyq_config['tag_on_oob'] = True
        pyq_config['sim_speedup_factor'] = 4  # Increase simulation speed
        
        # Determine rendering mode
        render_mode = 'human' if config.get('render', False) else None
        
        # Set up reward configuration for each agent
        # We'll use the caps_and_grabs reward function for all our team agents
        team_size = config.get('num_agents', 3)
        self.reward_config = {}
        for i in range(team_size):
            self.reward_config[f'agent_{i}'] = rewards.caps_and_grabs
        
        # Set None for opponent agents
        for i in range(team_size, team_size * 2):
            self.reward_config[f'agent_{i}'] = None
        
        # Create the Pyquaticus environment
        self.env = pyquaticus_v0.PyQuaticusEnv(
            config_dict=pyq_config,
            render_mode=render_mode,
            reward_config=self.reward_config,
            team_size=team_size
        )
        
        # Track environment state
        self.step_count = 0
        self.max_steps = config.get('max_steps', 1000)
        self.team_scores = [0, 0]
        self.map_size = np.array(pyq_config['env_bounds'])
        self.observation = None
        
        # Initialize environment
        self.reset()
    
    def reset(self) -> Dict[str, Any]:
        """Reset the environment and return initial observation."""
        self.env.reset(seed=None)
        self.step_count = 0
        self.team_scores = [0, 0]
        
        # Get observation from first agent
        obs = self.env.observe('agent_0')
        
        # Convert observation to format expected by trainer
        self.observation = self._format_observation(obs)
        return self.observation
    
    def step(self, action) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action for our agent (team 0, agent 0)
            
        Returns:
            observation: Updated observation
            reward: Reward from the step
            done: Whether the episode is done
            info: Additional information
        """
        # Convert action to Pyquaticus format (discretize if necessary)
        pyq_action = self._format_action(action)
        
        # Create action dictionary for all agents
        action_dict = {'agent_0': pyq_action}
        
        # Step the environment
        self.env.step(action_dict)
        
        # Get observation, reward, termination status for our agent
        obs = self.env.observe('agent_0')
        reward = self.env.get_rewards()['agent_0'] if 'agent_0' in self.env.get_rewards() else 0.0
        done = self.env.terminations['agent_0']
        
        # Update step count
        self.step_count += 1
        
        # Update team scores from Pyquaticus environment
        self.team_scores = [
            self.env.par_env.env.blue_score,
            self.env.par_env.env.red_score
        ]
        
        # Create info dictionary
        info = {
            'step': self.step_count,
            'team_scores': self.team_scores,
            'done_reason': 'time_limit' if self.step_count >= self.max_steps else 'game_end'
        }
        
        # Format observation for trainer
        self.observation = self._format_observation(obs)
        
        # Check if done due to max steps
        if self.step_count >= self.max_steps:
            done = True
        
        return self.observation, reward, done, info
    
    def _format_observation(self, obs) -> Dict[str, Any]:
        """
        Convert Pyquaticus observation to format expected by trainer.
        
        Args:
            obs: Pyquaticus observation
            
        Returns:
            formatted_obs: Observation in GameEnvironment format
        """
        # Extract info from Pyquaticus observation
        agent_obs = obs
        
        # For now, create a simplified observation with agent positions and flags
        # This can be expanded based on what the trainer expects
        agents_data = []
        for i in range(self.env.par_env.team_size * 2):  # Both teams
            agent_id = f'agent_{i}'
            team = 0 if i < self.env.par_env.team_size else 1
            
            # Try to get position from observation
            if hasattr(self.env.par_env.env, 'agents') and i < len(self.env.par_env.env.agents):
                position = self.env.par_env.env.agents[i].position
                velocity = self.env.par_env.env.agents[i].velocity
                is_tagged = self.env.par_env.env.agents[i].is_tagged
                has_flag = self.env.par_env.env.agents[i].has_flag
            else:
                # Use default values if not available
                position = np.zeros(2)
                velocity = np.zeros(2)
                is_tagged = False
                has_flag = False
            
            agents_data.append({
                'id': i,
                'position': position,
                'velocity': velocity,
                'has_flag': has_flag,
                'is_tagged': is_tagged,
                'team': team
            })
        
        # Get flag information
        flags_data = []
        for i in range(2):  # Blue and Red flags
            if hasattr(self.env.par_env.env, 'flags') and i < len(self.env.par_env.env.flags):
                position = self.env.par_env.env.flags[i].position
                is_captured = self.env.par_env.env.flags[i].is_captured
            else:
                position = np.zeros(2)
                is_captured = False
                
            flags_data.append({
                'position': position,
                'is_captured': is_captured,
                'team': i
            })
        
        # Create observation dictionary
        formatted_obs = {
            'agents': agents_data,
            'flags': flags_data,
            'team_scores': self.team_scores,
            'step': self.step_count,
            'map_size': self.map_size
        }
        
        return formatted_obs
    
    def _format_action(self, action):
        """
        Convert trainer action to Pyquaticus action format.
        
        Args:
            action: Action from trainer
            
        Returns:
            pyq_action: Action in Pyquaticus format
        """
        # Check if action is already in Pyquaticus format (discrete)
        if isinstance(action, (int, np.integer)):
            return action
        
        # For continuous actions, convert to Pyquaticus discrete format
        # Pyquaticus uses discrete actions represented by integers
        # The action space is defined in pyquaticus.config.ACTION_MAP
        
        # Extract speed and heading from continuous action
        # action[0] is typically speed, action[1] is heading in radians
        speed = np.clip(action[0], -1.0, 1.0)
        heading = np.clip(action[1], -1.0, 1.0) * 180  # Convert to degrees
        
        # Find the closest action in Pyquaticus ACTION_MAP
        action_map = pyquaticus.config.ACTION_MAP
        closest_action = 0
        min_distance = float('inf')
        
        for i, (map_speed, map_heading) in enumerate(action_map):
            # Skip the stop action
            if map_speed == 0.0:
                continue
                
            # Adjust map_speed to be in [-1, 1] range for comparison
            adjusted_speed = map_speed if map_speed <= 1.0 else 1.0
            
            # Calculate distance to this action
            speed_diff = abs(speed - adjusted_speed)
            heading_diff = min(abs(heading - map_heading), 360 - abs(heading - map_heading))
            
            # Weight speed more than heading
            distance = speed_diff * 3 + heading_diff / 180
            
            if distance < min_distance:
                min_distance = distance
                closest_action = i
        
        # Use stop action if speed is very low
        if abs(speed) < 0.2:
            closest_action = len(action_map) - 1  # Last action is stop
            
        return closest_action
    
    def render(self, mode='human'):
        """Render the environment."""
        if self.env.render_mode is not None:
            self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close() 