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
        self.last_env_info = None # Add field to store env info

        # Store team base locations from config (or defaults)
        default_bases = [np.array([0, 50]), np.array([100, 50])]
        self.env_team_bases = [
            np.array(base) for base in pyq_config.get('team_bases', default_bases)
        ]

        # Initialize environment
        self.reset()
    
    def reset(self) -> Dict[str, Any]:
        """Reset the environment and return initial observation."""
        initial_obs_dict, initial_info_dict = self.env.reset(seed=None) # Capture reset return values
        self.step_count = 0
        self.team_scores = [0, 0]
        self.last_env_info = initial_info_dict # Store env info

        # Get observation from the returned dictionary
        obs = initial_obs_dict['agent_0']

        # Convert observation to format expected by trainer, passing env info
        self.observation = self._format_observation(obs, self.last_env_info)
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
        # TODO: Extend this for multi-agent control if needed
        action_dict = {'agent_0': pyq_action}
        
        # Step the environment and capture return values
        obs_dict, reward_dict, termination_dict, truncation_dict, info_dict = self.env.step(action_dict)
        self.last_env_info = info_dict # Store env info
        
        # Get observation, reward, termination status for our agent from returned dicts
        obs = obs_dict.get('agent_0')
        reward = reward_dict.get('agent_0', 0.0)
        terminated = termination_dict.get('agent_0', False)
        truncated = truncation_dict.get('agent_0', False) # PettingZoo uses truncated for time limits etc.
        
        # Update step count
        self.step_count += 1
        
        # Update team scores from the environment info dictionary
        self.team_scores = [
            self.last_env_info.get('blue_score', self.team_scores[0]), # Get from info dict
            self.last_env_info.get('red_score', self.team_scores[1])   # Get from info dict
        ]
        
        # Create info dictionary - merge internal info with env info?
        info = {
            'step': self.step_count,
            'team_scores': self.team_scores,
            'done_reason': 'time_limit' if self.step_count >= self.max_steps else 'game_end',
            'env_info': info_dict # Include original info dict
        }
        
        # Format observation for trainer, passing env info
        self.observation = self._format_observation(obs, self.last_env_info)
        
        # Check if done due to termination, truncation, or max steps
        done = terminated or truncated or (self.step_count >= self.max_steps)
        
        return self.observation, reward, done, info
    
    def _format_observation(self, obs, env_info) -> Dict[str, Any]:
        """
        Convert Pyquaticus observation to format expected by trainer.
        Uses the env_info dict for global state.
        
        Args:
            obs: Pyquaticus observation for the specific agent (potentially unused if info has all)
            env_info: Info dictionary from the environment step/reset
            
        Returns:
            formatted_obs: Observation in GameEnvironment format
        """
        # Extract info from Pyquaticus observation (agent's perspective)
        agent_obs = obs
        
        # Use env_info dictionary to get global state (agent and flag positions/status)
        agents_data = []
        team_size = self.config.get('num_agents', 3) # Get team size from config
        env_agents = env_info.get('agents') # Get agent states from info dict

        for i in range(team_size * 2):  # Both teams
            agent_id = f'agent_{i}'
            team = 0 if i < team_size else 1

            # Default values
            position = np.zeros(2)
            velocity = np.zeros(2)
            is_tagged = False
            has_flag = False

            # Try to get state from env_info['agents'] list
            # Assuming env_agents is a list ordered by agent index
            if env_agents and i < len(env_agents):
                agent_state = env_agents[i]
                # Check if agent_state is an object or dict
                if hasattr(agent_state, 'position'):
                    position = agent_state.position
                    velocity = agent_state.velocity
                    is_tagged = agent_state.is_tagged
                    has_flag = agent_state.has_flag
                elif isinstance(agent_state, dict):
                    position = agent_state.get('position', position)
                    velocity = agent_state.get('velocity', velocity)
                    is_tagged = agent_state.get('is_tagged', is_tagged)
                    has_flag = agent_state.get('has_flag', has_flag)

            # Calculate health based on tagged status
            health = 0.0 if is_tagged else 1.0

            agents_data.append({
                'id': i,
                'position': np.array(position), # Ensure numpy array
                'velocity': np.array(velocity), # Ensure numpy array
                'has_flag': has_flag,
                'is_tagged': is_tagged,
                'health': health, # Add health key
                'team': team
            })

        # Get flag information from env_info
        flags_data = []
        env_flags = env_info.get('flags') # Get flag states from info dict
        for i in range(2):  # Blue and Red flags
            # Default values
            position = np.zeros(2)
            is_captured = False

            # Try to get state from env_info['flags'] list
            # Assuming env_flags is a list ordered by team index (0=Blue, 1=Red)
            if env_flags and i < len(env_flags):
                flag_state = env_flags[i]
                if hasattr(flag_state, 'position'):
                    position = flag_state.position
                    is_captured = flag_state.is_captured
                elif isinstance(flag_state, dict):
                    position = flag_state.get('position', position)
                    is_captured = flag_state.get('is_captured', is_captured)

            flags_data.append({
                'position': np.array(position), # Ensure numpy array
                'is_captured': is_captured,
                'team': i
            })
        
        # Get team base locations directly from the environment object
        team_bases = None
        if hasattr(self.env, 'team_bases'):
            team_bases = self.env.team_bases

        if team_bases is None:
            # Use default positions if not found on env object
            team_bases = [np.array([0, 50]), np.array([100, 50])]
            if self.debug_level >= 1:
                print("WARN: 'team_bases' not found on self.env. Using defaults.")
        else:
            # Ensure numpy arrays
            team_bases = [np.array(base) for base in team_bases]

        # Create observation dictionary
        formatted_obs = {
            'agents': agents_data,
            'flags': flags_data,
            'team_scores': self.team_scores,
            'team_bases': team_bases, # Add team base positions
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