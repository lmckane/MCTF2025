import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from collections import deque
import random
from hrl.utils.state_processor import ProcessedState
from hrl.utils.experience import Experience, ReplayBuffer

class PolicyNetwork(nn.Module):
    """Neural network for the policy."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_size * 2)  # Mean and log_std for each action dimension
        self.value_head = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Get policy parameters
        policy_params = self.policy_head(x)
        mean = policy_params[:, :self.action_size]
        log_std = policy_params[:, self.action_size:]
        std = torch.exp(log_std)
        
        # Create normal distribution
        policy = torch.distributions.Normal(mean, std)
        
        # Get value
        value = self.value_head(x)
        return policy, value

class HierarchicalPolicy:
    """Hierarchical policy for the capture-the-flag game."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the hierarchical policy."""
        self.config = config
        self.state_size = 8  # Position[2], velocity[2], flags[1], tags[1], health[1], team[1]
        self.action_size = config['action_size']
        self.hidden_size = config['hidden_size']
        self.options = config['options']
        self.learning_rate = config['learning_rate']
        self.gamma = config['gamma']
        self.lambda_ = config['lambda_']
        self.entropy_coef = config['entropy_coef']
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.option_networks = {}
        for option in self.options:
            network = PolicyNetwork(self.state_size, self.action_size, self.hidden_size)
            network.to(self.device)
            self.option_networks[option] = network
            
        # Initialize optimizers
        self.optimizers = {}
        for option in self.options:
            optimizer = torch.optim.Adam(self.option_networks[option].parameters(), lr=self.learning_rate)
            self.optimizers[option] = optimizer
            
        # Initialize replay buffer
        self.buffer = ReplayBuffer()
        
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
        state_tensor = torch.FloatTensor(features).to(self.device)
        return state_tensor
        
    def get_action(self, state: ProcessedState, option: str) -> np.ndarray:
        """Get action from policy for given state and option."""
        # Convert ProcessedState to tensor
        state_tensor = torch.tensor([
            state.agent_positions[0][0], state.agent_positions[0][1],
            state.agent_velocities[0][0], state.agent_velocities[0][1],
            state.agent_flags[0],
            state.agent_tags[0],
            state.agent_health[0],
            state.agent_teams[0]
        ], dtype=torch.float32).unsqueeze(0)
        
        # Get action distribution from option network
        policy, _ = self.option_networks[option](state_tensor)
        
        # Sample action and clip to valid range
        action = policy.sample()
        action = torch.clamp(action, -1, 1)  # Clip to [-1, 1] range
        return action.squeeze(0).numpy()  # Remove batch dimension and convert to numpy
            
    def update(self, experiences: List[Experience], advantages: torch.Tensor):
        """Update policy using PPO."""
        # Group experiences by option
        option_experiences = {}
        for exp in experiences:
            if exp.option not in option_experiences:
                option_experiences[exp.option] = []
            option_experiences[exp.option].append(exp)
            
        # Update each option's policy
        for option, option_exps in option_experiences.items():
            if not option_exps:
                continue
                
            # Convert states to tensors
            def state_to_tensor(state: ProcessedState) -> torch.Tensor:
                return torch.tensor([
                    state.agent_positions[0][0], state.agent_positions[0][1],
                    state.agent_velocities[0][0], state.agent_velocities[0][1],
                    state.agent_flags[0],
                    state.agent_tags[0],
                    state.agent_health[0],
                    state.agent_teams[0]
                ], dtype=torch.float32)
            
            # Get tensors from experiences
            states = torch.stack([state_to_tensor(exp.processed_state) for exp in option_exps])
            actions = torch.stack([torch.tensor(exp.action, dtype=torch.float32) for exp in option_exps])
            rewards = torch.tensor([exp.reward for exp in option_exps], dtype=torch.float32)
            
            # Get current policy and value predictions
            policy, values = self.option_networks[option](states)
            
            # Calculate policy loss - ensure consistent dimensions
            log_probs = policy.log_prob(actions).sum(dim=1)  # Sum across action dimensions
            option_advantages = advantages[:len(option_exps)]  # Slice advantages to match batch size
            policy_loss = -(log_probs * option_advantages)
            
            # Calculate value loss
            if len(option_exps) > 1:  # Only compute value loss if we have enough samples
                # Ensure consistent dimensions by using view() to reshape tensors
                values_slice = values[:-1].view(-1)
                # Create value targets with consistent shape
                value_targets = (rewards[:-1] + self.gamma * values[1:].view(-1)).view(-1)
                # Now both tensors should have same shape [batch_size-1]
                value_loss = F.mse_loss(values_slice, value_targets)
            else:
                # If only one sample, skip value loss
                value_loss = torch.tensor(0.0).to(self.device)
            
            # Calculate entropy bonus
            entropy = policy.entropy().mean()
            
            # Total loss
            loss = policy_loss.mean() + 0.5 * value_loss - self.entropy_coef * entropy
            
            # Update network
            self.optimizers[option].zero_grad()
            loss.backward()
            self.optimizers[option].step()
            
    def update_adversarial(self, experiences: List[Experience]):
        """Update policy using adversarial experiences."""
        # Similar to regular update but with different reward scaling
        advantages = torch.FloatTensor([exp.reward * 2 for exp in experiences]).to(self.device)
        self.update(experiences, advantages)
        
    def get_q_value(self, processed_state: torch.Tensor, action: np.ndarray, option: int) -> float:
        """Get Q-value for state-action pair."""
        with torch.no_grad():
            # Make sure processed_state has batch dimension
            if processed_state.dim() == 1:
                processed_state = processed_state.unsqueeze(0)
                
            # Get value
            _, value = self.option_networks[option](processed_state)
            return value.item()
            
    def get_advantage(self, processed_state: torch.Tensor, action: np.ndarray, option: int) -> float:
        """Get advantage for state-action pair."""
        with torch.no_grad():
            # Make sure processed_state has batch dimension
            if processed_state.dim() == 1:
                processed_state = processed_state.unsqueeze(0)
                
            # Get policy and value
            policy, value = self.option_networks[option](processed_state)
            
            # Make sure action has correct shape
            action_tensor = torch.tensor(action, dtype=torch.float32).to(self.device)
            if action_tensor.dim() == 1:
                action_tensor = action_tensor.unsqueeze(0)
                
            # Calculate log probability
            log_prob = policy.log_prob(action_tensor).sum(dim=1)
            
            # Calculate advantage
            q_value = self.get_q_value(processed_state, action, option)
            advantage = log_prob.exp() * (value.item() - q_value)
            
            return advantage.item()
            
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for saving."""
        return {
            'option_networks': {option: network.state_dict() for option, network in self.option_networks.items()},
            'config': self.config
        }
        
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state from dictionary."""
        self.config = state_dict['config']
        for option, state in state_dict['option_networks'].items():
            if option in self.option_networks:
                self.option_networks[option].load_state_dict(state) 