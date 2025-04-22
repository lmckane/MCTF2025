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
        # Convert state to tensor
        state_tensor = torch.tensor([
            state.agent_positions[0][0], state.agent_positions[0][1],
            state.agent_velocities[0][0], state.agent_velocities[0][1],
            state.agent_flags[0],
            state.agent_tags[0],
            state.agent_health[0],
            state.agent_teams[0]
        ], dtype=torch.float32).unsqueeze(0)
        
        # Get option scores
        option_scores = self.network(state_tensor)
        
        # Apply option weights
        weighted_scores = option_scores * torch.tensor([self.option_weights[option] for option in self.config['options']])
        
        # Select option with highest score
        option_idx = torch.argmax(weighted_scores).item()
        return self.config['options'][option_idx]
        
    def update_weights(self, experiences: List[Experience]):
        """Update option weights based on experiences."""
        for exp in experiences:
            # Get option score
            state_tensor = torch.tensor([
                exp.processed_state.agent_positions[0][0], exp.processed_state.agent_positions[0][1],
                exp.processed_state.agent_velocities[0][0], exp.processed_state.agent_velocities[0][1],
                exp.processed_state.agent_flags[0],
                exp.processed_state.agent_tags[0],
                exp.processed_state.agent_health[0],
                exp.processed_state.agent_teams[0]
            ], dtype=torch.float32).unsqueeze(0)
            
            option_scores = self.network(state_tensor)
            option_idx = self.config['options'].index(exp.option)
            
            # Update weight based on reward
            self.option_weights[exp.option] += exp.reward * 0.01  # Small learning rate
            
            # Normalize weights
            total_weight = sum(self.option_weights.values())
            for option in self.option_weights:
                self.option_weights[option] /= total_weight
        
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
            'config': self.config
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state from dictionary."""
        self.network.load_state_dict(state_dict['network'])
        self.option_weights = state_dict['option_weights']
        self.config = state_dict['config'] 