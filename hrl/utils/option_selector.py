import numpy as np
from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from hrl.utils.state_processor import ProcessedState
from hrl.policies.hierarchical_policy import Experience

class OptionSelector:
    """Selects options based on state."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the option selector."""
        self.config = config
        self.state_size = 8  # Position[2], velocity[2], flags[1], tags[1], health[1], team[1]
        self.num_options = len(config['options'])
        self.hidden_size = config.get('hidden_size', 128)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.debug_level = config.get('debug_level', 1)  # Default to minimal debugging
        
        # Initialize network
        self.network = nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_options)
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
        # Initialize option weights
        self.option_weights = {option: 1.0 for option in config['options']}
        
    def select_option(self, state: ProcessedState) -> str:
        """Select an option based on state."""
        try:
            # Basic defensive check - use first agent (controlled agent) data
            if not hasattr(state, 'agent_positions') or len(state.agent_positions) == 0:
                if self.debug_level >= 1:
                    print("Warning: State missing agent_positions, using default option")
                return self.config['options'][0]  # Default to first option
                
            # Convert state to tensor
            state_tensor = torch.tensor([
                state.agent_positions[0][0], state.agent_positions[0][1],
                state.agent_velocities[0][0], state.agent_velocities[0][1],
                float(state.agent_flags[0]),
                float(state.agent_tags[0]),
                state.agent_health[0],
                float(state.agent_teams[0])
            ], dtype=torch.float32).unsqueeze(0)
            
            # Get option scores
            option_scores = self.network(state_tensor)
            
            # Apply option weights
            option_weights_tensor = torch.tensor([self.option_weights[option] for option in self.config['options']])
            weighted_scores = option_scores * option_weights_tensor
            
            # Print debug info (very rarely)
            if self.debug_level >= 2 and np.random.random() < 0.001:  # Reduced from 1% to 0.1%
                print("\nOption Selection Debug:")
                print(f"  State: pos={state.agent_positions[0]}, vel={state.agent_velocities[0]}, flag={state.agent_flags[0]}, tag={state.agent_tags[0]}")
                print("  Option scores:")
                for i, option in enumerate(self.config['options']):
                    print(f"    {option}: score={option_scores[0][i].item():.2f}, weight={self.option_weights[option]:.2f}, weighted={weighted_scores[0][i].item():.2f}")
            
            # Select option with highest score
            option_idx = torch.argmax(weighted_scores).item()
            selected_option = self.config['options'][option_idx]
            
            return selected_option
        except Exception as e:
            if self.debug_level >= 1:
                print(f"Error in select_option: {e}")
            return self.config['options'][0]  # Default to first option if error occurs
        
    def update_weights(self, experiences: List[Experience]):
        """Update option weights based on experiences."""
        if not experiences:
            return
            
        try:
            option_rewards = {}
            option_counts = {}
            
            # Calculate average reward for each option
            for exp in experiences:
                if not hasattr(exp, 'option') or not hasattr(exp, 'reward'):
                    continue
                    
                option = exp.option
                if option not in option_rewards:
                    option_rewards[option] = 0.0
                    option_counts[option] = 0
                    
                option_rewards[option] += exp.reward
                option_counts[option] += 1
            
            # Update weights based on average rewards
            for option, reward in option_rewards.items():
                if option_counts[option] > 0:
                    avg_reward = reward / option_counts[option]
                    # Use a small learning rate and ensure reward is having an impact
                    update_factor = 1.0 + avg_reward * 0.01  # Small learning rate
                    self.option_weights[option] *= update_factor
            
            # Normalize weights
            total_weight = sum(self.option_weights.values())
            if total_weight > 0:  # Avoid division by zero
                for option in self.option_weights:
                    self.option_weights[option] /= total_weight
                    
            # Occasionally print weights for debugging (reduced frequency)
            if self.debug_level >= 2 and np.random.random() < 0.001:  # Reduced from 1% to 0.1%
                print("\nOption weights updated:")
                for option, weight in self.option_weights.items():
                    print(f"  {option}: {weight:.4f}")
                    
        except Exception as e:
            if self.debug_level >= 1:
                print(f"Error in update_weights: {e}")
        
    def process_state(self, state: ProcessedState) -> torch.Tensor:
        """Convert processed state to tensor."""
        # Extract relevant features
        features = []
        
        # Add agent features
        features.extend(state.agent_positions.flatten())  # Shape: (num_agents * 2,)
        features.extend(state.agent_velocities.flatten())  # Shape: (num_agents * 2,)
        features.extend(state.agent_flags)  # Shape: (num_agents,)
        features.extend(state.agent_tags)  # Shape: (num_agents,)
        features.extend(state.agent_teams)  # Shape: (num_agents,)
        features.extend(state.agent_health)  # Shape: (num_agents,)
        
        # Add flag features
        features.extend(state.flag_positions.flatten())  # Shape: (num_flags * 2,)
        features.extend(state.flag_captured)  # Shape: (num_flags,)
        features.extend(state.flag_teams)  # Shape: (num_flags,)
        
        # Add base positions
        features.extend(state.base_positions.flatten())  # Shape: (num_teams * 2,)
        
        # Add game state
        features.append(state.step_count / 1000.0)  # Normalize step count
        features.append(state.game_state)
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(features).unsqueeze(0)
        return state_tensor
        
    def update(self, state: Dict[str, Any], reward: float, done: bool):
        """Update option selector based on reward."""
        if done:
            # Update weights
            self.update_weights(state['experiences'])
            
    def train(self, states: List[Dict[str, Any]], 
             options: List[int], rewards: List[float]):
        """Train the option selector network."""
        # Convert states to tensors
        state_tensors = torch.stack([self.process_state(state) for state in states])
        option_tensors = torch.LongTensor(options).unsqueeze(1)
        reward_tensors = torch.FloatTensor(rewards).unsqueeze(1)
        
        # Get option scores
        option_scores = self.network(state_tensors)
        
        # Calculate loss (cross entropy with reward weighting)
        loss = F.cross_entropy(option_scores, option_tensors, reduction='none')
        loss = (loss * reward_tensors).mean()
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for saving."""
        return {
            'network': self.network.state_dict(),
            'option_weights': self.option_weights,
            'config': self.config,
            'debug_level': self.debug_level
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state from dictionary."""
        self.network.load_state_dict(state_dict['network'])
        self.option_weights = state_dict['option_weights']
        self.config = state_dict['config']
        if 'debug_level' in state_dict:
            self.debug_level = state_dict['debug_level'] 