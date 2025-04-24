"""
PRIMARY OPTION SELECTOR IMPLEMENTATION
This is the canonical implementation of the OptionSelector to be used throughout the codebase.
This version uses a neural network with role-based coordination for option selection.

Other implementations in option_selection.py and policies/option_selector.py are deprecated
and should be refactored to use this implementation.
"""

import numpy as np
from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from hrl.utils.state_processor import ProcessedState
from hrl.utils.experience import Experience
from hrl.utils.team_coordinator import AgentRole

class OptionSelector:
    """Selects options based on state with team coordination awareness."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the option selector."""
        self.config = config
        # Extended state size to include role and coordination data
        self.state_size = 21  # Updated from 16 to match actual input dimensions
        self.num_options = len(config['options'])
        self.hidden_size = config.get('hidden_size', 128)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.debug_level = config.get('debug_level', 1)  # Default to minimal debugging
        
        # Role-specific option weights (initialized equally)
        self.role_option_weights = {
            AgentRole.ATTACKER.value: {option: 1.0 for option in config['options']},
            AgentRole.DEFENDER.value: {option: 1.0 for option in config['options']},
            AgentRole.INTERCEPTOR.value: {option: 1.0 for option in config['options']}
        }
        
        # Initialize network with larger capacity for team information
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
        """Select an option based on state and team coordination."""
        try:
            # Basic defensive check
            if not hasattr(state, 'agent_positions') or len(state.agent_positions) == 0:
                if self.debug_level >= 1:
                    print("Warning: State missing agent_positions, using default option")
                return self.config['options'][0]  # Default to first option
                
            # Preprocessing for role-specific behavior
            available_options = self.config['options']
            
            has_recommended_target = hasattr(state, 'recommended_target') and state.recommended_target is not None and len(state.recommended_target) >= 2
            
            # Determine agent role and target distance
            agent_role = int(state.agent_roles[0]) if hasattr(state, 'agent_roles') and len(state.agent_roles) > 0 else AgentRole.ATTACKER.value
            
            # Calculate distances for coordination
            target_distance = 0.0
            team_spread = 0.0
            
            if has_recommended_target:
                # Calculate distance to recommended target
                agent_pos = np.array([state.agent_positions[0][0], state.agent_positions[0][1]])
                target_pos = np.array([state.recommended_target[0], state.recommended_target[1]])
                target_distance = np.linalg.norm(target_pos - agent_pos)
            
            # Calculate team spread (distance between team agents)
            if hasattr(state, 'agent_teams') and hasattr(state, 'agent_positions'):
                team_agents = []
                for i in range(len(state.agent_teams)):
                    if state.agent_teams[i] == state.agent_teams[0]:  # Same team
                        team_agents.append(np.array([state.agent_positions[i][0], state.agent_positions[i][1]]))
                
                if len(team_agents) > 1:
                    dists = []
                    for i in range(len(team_agents)):
                        for j in range(i+1, len(team_agents)):
                            dists.append(np.linalg.norm(team_agents[i] - team_agents[j]))
                    team_spread = np.mean(dists) if dists else 0.0
            
            # Get distance to nearest enemy and check if enemies are tagged
            enemy_distance = 1.0  # Default to normalized max distance
            tagged_enemies = 0
            total_enemies = 0
            for i in range(len(state.agent_positions)):
                if i > 0 and state.agent_teams[i] != state.agent_teams[0]:  # Enemy team
                    total_enemies += 1
                    if state.agent_tags[i] > 0:  # Enemy is tagged
                        tagged_enemies += 1
                    dist = np.linalg.norm(state.agent_positions[0] - state.agent_positions[i])
                    enemy_distance = min(enemy_distance, dist)
            
            # Calculate enemy flag and our flag distance
            enemy_flag_pos = None
            our_flag_pos = None
            for i in range(len(state.flag_positions)):
                if state.flag_teams[i] != state.agent_teams[0]:  # Enemy flag
                    enemy_flag_pos = state.flag_positions[i]
                else:  # Our flag
                    our_flag_pos = state.flag_positions[i]
            
            enemy_flag_distance = np.linalg.norm(state.agent_positions[0] - enemy_flag_pos) if enemy_flag_pos is not None else 1.0
            our_flag_distance = np.linalg.norm(state.agent_positions[0] - our_flag_pos) if our_flag_pos is not None else 1.0
            
            # Check if we have opponent modeling data and counter strategy
            counter_strategy = getattr(state, 'counter_strategy', "balanced")
            position_danger = getattr(state, 'position_danger', 0.0)
            
            # Create input tensor with enhanced coordination features
            # Ensure all values are properly converted to scalars when needed
            input_features = []
            
            # Basic agent features
            try:
                # Extract and ensure all values are scalars
                pos_x = self._safe_scalar(state.agent_positions[0][0])
                pos_y = self._safe_scalar(state.agent_positions[0][1])
                vel_x = self._safe_scalar(state.agent_velocities[0][0])
                vel_y = self._safe_scalar(state.agent_velocities[0][1])
                
                input_features.extend([
                    pos_x, pos_y,
                    vel_x, vel_y,
                    self._safe_scalar(state.agent_flags[0]),
                    self._safe_scalar(state.agent_tags[0]),
                    self._safe_scalar(state.agent_health[0]),
                    self._safe_scalar(state.agent_teams[0])
                ])
                
                # Make sure agent_role is also safely converted if it's an array
                agent_role = self._safe_scalar(state.agent_roles[0])
                
                # Role and coordination features
                input_features.append(agent_role)
                
                # Safely handle target distance
                if has_recommended_target:
                    target_dist = self._safe_scalar(target_distance)
                    input_features.append(target_dist)
                else:
                    input_features.append(0.0)
                
                # Safely handle recommended target position
                if has_recommended_target:
                    rec_x = self._safe_scalar(state.recommended_target[0])
                    rec_y = self._safe_scalar(state.recommended_target[1])
                    input_features.extend([rec_x, rec_y])
                else:
                    input_features.extend([0.0, 0.0])
                    
                # Additional flags and metrics - ensure they're all scalars
                our_flag_threat = self._safe_scalar(state.our_flag_threat) if hasattr(state, 'our_flag_threat') else 0.0
                our_flag_captured = self._safe_scalar(state.our_flag_captured) if hasattr(state, 'our_flag_captured') else 0.0
                enemy_flag_captured = self._safe_scalar(state.enemy_flag_captured) if hasattr(state, 'enemy_flag_captured') else 0.0
                team_spread_val = self._safe_scalar(team_spread)
                
                input_features.append(our_flag_threat)
                input_features.append(our_flag_captured)
                input_features.append(enemy_flag_captured)
                input_features.append(team_spread_val)
                
                # Add team spread (duplicate - keeping for compatibility)
                input_features.append(team_spread_val)
                
                # Add nearest enemy if available
                nearest_enemy_dist = 1000.0  # Default large value
                nearest_enemy_tagged = 0.0
                
                if hasattr(state, 'enemy_positions') and len(state.enemy_positions) > 0:
                    # Find nearest enemy
                    agent_pos = np.array([pos_x, pos_y])
                    enemy_dists = []
                    
                    for i in range(len(state.enemy_positions)):
                        try:
                            enemy_pos = np.array([
                                self._safe_scalar(state.enemy_positions[i][0]),
                                self._safe_scalar(state.enemy_positions[i][1])
                            ])
                            dist = np.linalg.norm(agent_pos - enemy_pos)
                            enemy_dists.append((dist, i))
                        except Exception as e:
                            if self.debug_level >= 2:
                                print(f"Error processing enemy position {i}: {e}")
                            continue
                    
                    if enemy_dists:
                        enemy_dists.sort()
                        nearest_enemy_dist = float(enemy_dists[0][0])
                        enemy_idx = enemy_dists[0][1]
                        
                        # Check if enemy is tagged
                        if hasattr(state, 'enemy_tags') and len(state.enemy_tags) > enemy_idx:
                            enemy_tag = self._safe_scalar(state.enemy_tags[enemy_idx])
                            nearest_enemy_tagged = float(enemy_tag)
                
                input_features.append(nearest_enemy_dist)
                input_features.append(nearest_enemy_tagged)
                
                # Add game context information if available
                if hasattr(state, 'time_remaining'):
                    time_remaining = self._safe_scalar(state.time_remaining)
                    input_features.append(time_remaining)
                else:
                    input_features.append(0.0)
                
                if hasattr(state, 'score_diff'):
                    score_diff = self._safe_scalar(state.score_diff)
                    input_features.append(score_diff)
                else:
                    input_features.append(0.0)
                
                # Convert to tensor for model input
                input_tensor = self._safe_tensor_creation(input_features)
                
                # Get model predictions
                outputs = self._safe_forward(input_tensor)
                if outputs is None:
                    return self.config['options'][0]  # Default to first option if error
                    
                # Apply role-based weights and filtering
                weighted_outputs = outputs[0].clone()
                
                # Adjust outputs based on agent role
                for option_idx, option_name in enumerate(self.config['options']):
                    role_weight = self._get_role_weight(agent_role, option_name)
                    weighted_outputs[option_idx] *= role_weight
                    
                    # Apply conditional filters
                    if not self._check_option_conditions(agent_role, option_name, state):
                        weighted_outputs[option_idx] = -float('inf')  # Effectively disable this option
                
                # Filter to only available options
                available_indices = [self.config['options'].index(option) for option in available_options]
                for i in range(len(weighted_outputs)):
                    if i not in available_indices:
                        weighted_outputs[i] = -float('inf')
                
                # Select based on highest weighted score
                selected_idx = torch.argmax(weighted_outputs).item()
                selected_option = self.config['options'][selected_idx]
                
                return selected_option
            except Exception as e:
                if self.debug_level >= 1:
                    print(f"Error in enhanced select_option: {e}")
                return self.config['options'][0]  # Default to first option if error occurs
        except Exception as e:
            if self.debug_level >= 1:
                print(f"Error in enhanced select_option: {e}")
            return self.config['options'][0]  # Default to first option if error occurs
        
    def update_weights(self, experiences: List[Experience]):
        """Update option weights based on experiences, now considering agent roles."""
        if not experiences:
            return
            
        try:
            # Track rewards by role and option
            role_option_rewards = {}
            role_option_counts = {}
            
            # General option rewards (role-agnostic)
            option_rewards = {}
            option_counts = {}
            
            # Calculate average reward for each option and role
            for exp in experiences:
                if not hasattr(exp, 'option') or not hasattr(exp, 'reward') or not hasattr(exp, 'processed_state'):
                    continue
                    
                option = exp.option
                
                # Get agent role from processed state
                state = exp.processed_state
                agent_role = int(state.agent_roles[0]) if hasattr(state, 'agent_roles') and len(state.agent_roles) > 0 else AgentRole.ATTACKER.value
                
                # Update general option stats
                if option not in option_rewards:
                    option_rewards[option] = 0.0
                    option_counts[option] = 0
                option_rewards[option] += exp.reward
                option_counts[option] += 1
                
                # Update role-specific option stats
                if agent_role not in role_option_rewards:
                    role_option_rewards[agent_role] = {}
                    role_option_counts[agent_role] = {}
                
                if option not in role_option_rewards[agent_role]:
                    role_option_rewards[agent_role][option] = 0.0
                    role_option_counts[agent_role][option] = 0
                    
                role_option_rewards[agent_role][option] += exp.reward
                role_option_counts[agent_role][option] += 1
            
            # Update general weights based on average rewards
            for option, reward in option_rewards.items():
                if option_counts[option] > 0:
                    avg_reward = reward / option_counts[option]
                    # Use a small learning rate and ensure reward is having an impact
                    update_factor = 1.0 + avg_reward * 0.01  # Small learning rate
                    self.option_weights[option] *= update_factor
            
            # Update role-specific weights
            for role, rewards in role_option_rewards.items():
                for option, reward in rewards.items():
                    if role_option_counts[role][option] > 0:
                        avg_reward = reward / role_option_counts[role][option]
                        # Use a slightly larger learning rate for role-specific updates
                        update_factor = 1.0 + avg_reward * 0.02
                        if role in self.role_option_weights and option in self.role_option_weights[role]:
                            self.role_option_weights[role][option] *= update_factor
            
            # Normalize general weights
            total_weight = sum(self.option_weights.values())
            if total_weight > 0:  # Avoid division by zero
                for option in self.option_weights:
                    self.option_weights[option] /= total_weight
            
            # Normalize role-specific weights
            for role in self.role_option_weights:
                total_weight = sum(self.role_option_weights[role].values())
                if total_weight > 0:  # Avoid division by zero
                    for option in self.role_option_weights[role]:
                        self.role_option_weights[role][option] /= total_weight
                    
            # Occasionally print weights for debugging
            if self.debug_level >= 2 and np.random.random() < 0.001:  # 0.1% chance to print debug info
                print("\nOption weights updated (with coordination):")
                print("General weights:")
                for option, weight in self.option_weights.items():
                    print(f"  {option}: {weight:.4f}")
                    
                print("Role-specific weights:")
                for role in self.role_option_weights:
                    role_name = AgentRole(role).name if role in [r.value for r in AgentRole] else f"ROLE_{role}"
                    print(f"  {role_name}:")
                    for option, weight in self.role_option_weights[role].items():
                        print(f"    {option}: {weight:.4f}")
                    
        except Exception as e:
            if self.debug_level >= 1:
                print(f"Error in coordinated update_weights: {e}")
        
    def process_state(self, state: ProcessedState) -> torch.Tensor:
        """Convert processed state to tensor with team coordination features."""
        try:
            # Extract relevant features
            features = []
            
            # Add agent features (safely)
            if hasattr(state, 'agent_positions') and len(state.agent_positions) > 0:
                features.extend(state.agent_positions.flatten())  # Shape: (num_agents * 2,)
            else:
                features.extend([0.0, 0.0])  # Default positions
                
            if hasattr(state, 'agent_velocities') and len(state.agent_velocities) > 0:
                features.extend(state.agent_velocities.flatten())  # Shape: (num_agents * 2,)
            else:
                features.extend([0.0, 0.0])  # Default velocities
                
            # Add other agent features with safe access
            for attr in ['agent_flags', 'agent_tags', 'agent_teams', 'agent_health']:
                if hasattr(state, attr) and len(getattr(state, attr)) > 0:
                    features.extend(getattr(state, attr))
                else:
                    features.append(0.0)  # Default value
            
            # Add coordination features if available
            if hasattr(state, 'agent_roles') and len(state.agent_roles) > 0:
                features.extend(state.agent_roles)  # Shape: (num_agents,)
            else:
                features.append(0.0)  # Default role
            
            if hasattr(state, 'recommended_target') and len(state.recommended_target) >= 2:
                features.extend(state.recommended_target[:2])  # Shape: (2,)
            else:
                features.extend([0.0, 0.0])  # Default target
                
            # Add flag threat states
            for attr in ['our_flag_threat', 'our_flag_captured', 'enemy_flag_captured']:
                if hasattr(state, attr):
                    val = getattr(state, attr)
                    if isinstance(val, np.ndarray) and val.size > 0:
                        features.append(float(val.item()))
                    else:
                        features.append(float(val))
                else:
                    features.append(0.0)
            
            # Add flag features
            if hasattr(state, 'flag_positions') and len(state.flag_positions) > 0:
                features.extend(state.flag_positions.flatten()[:4])  # Limit to prevent over-extension
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])  # Default flag positions
                
            if hasattr(state, 'flag_captured') and len(state.flag_captured) > 0:
                features.extend(state.flag_captured[:2])  # Limit to prevent over-extension
            else:
                features.extend([0.0, 0.0])  # Default flag capture states
                
            if hasattr(state, 'flag_teams') and len(state.flag_teams) > 0:
                features.extend(state.flag_teams[:2])  # Limit to prevent over-extension
            else:
                features.extend([0.0, 0.0])  # Default flag teams
            
            # Add base positions
            if hasattr(state, 'base_positions') and len(state.base_positions) > 0:
                features.extend(state.base_positions.flatten()[:4])  # Limit to prevent over-extension
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])  # Default base positions
            
            # Add game state
            if hasattr(state, 'step_count'):
                features.append(state.step_count / 1000.0)  # Normalize step count
            else:
                features.append(0.0)
                
            if hasattr(state, 'game_state'):
                features.append(float(state.game_state))
            else:
                features.append(0.0)
            
            # Convert to tensor and ensure it has the right size
            state_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # Pad or truncate to match expected network input size
            if state_tensor.shape[1] < self.state_size:
                # Pad with zeros
                padding = torch.zeros(state_tensor.shape[0], self.state_size - state_tensor.shape[1])
                state_tensor = torch.cat([state_tensor, padding], dim=1)
            elif state_tensor.shape[1] > self.state_size:
                # Truncate
                state_tensor = state_tensor[:, :self.state_size]
                
            return state_tensor
        except Exception as e:
            if self.debug_level >= 1:
                print(f"Error in process_state: {e}")
            # Return zero tensor of the expected size
            return torch.zeros(1, self.state_size)
        
    def update(self, state: Dict[str, Any], reward: float, done: bool):
        """Update option selector based on reward."""
        if done:
            # Update weights
            self.update_weights(state['experiences'])
            
    def train(self, states: List[Dict[str, Any]], 
             options: List[int], rewards: List[float]):
        """Train the option selector network."""
        try:
            # Convert states to tensors
            state_tensors = []
            for state in states:
                tensor = self.process_state(state)
                # Check if tensor needs resizing
                if tensor.shape[1] != self.state_size:
                    if tensor.shape[1] < self.state_size:
                        # Pad with zeros
                        padding = torch.zeros(tensor.shape[0], self.state_size - tensor.shape[1])
                        tensor = torch.cat([tensor, padding], dim=1)
                    else:
                        # Truncate
                        tensor = tensor[:, :self.state_size]
                state_tensors.append(tensor)
            
            if not state_tensors:
                return  # No data to train on
                
            state_tensors = torch.cat(state_tensors)
            option_tensors = torch.LongTensor(options)
            reward_tensors = torch.FloatTensor(rewards)
            
            # Get option scores
            option_scores = self.network(state_tensors)
            
            # Calculate loss (cross entropy with reward weighting)
            loss = F.cross_entropy(option_scores, option_tensors, reduction='none')
            loss = (loss * reward_tensors).mean()
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        except Exception as e:
            if self.debug_level >= 1:
                print(f"Error in train method: {e}")
        
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for saving."""
        return {
            'network': self.network.state_dict(),
            'option_weights': self.option_weights,
            'role_option_weights': self.role_option_weights,
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
        if 'role_option_weights' in state_dict:
            self.role_option_weights = state_dict['role_option_weights'] 

    def _get_role_weight(self, agent_role, option_name):
        """Get the weight for this option based on agent role"""
        if not hasattr(self, 'role_option_weights'):
            return 1.0
        
        # Convert agent_role to int using our safe method
        try:
            agent_role = int(self._safe_scalar(agent_role))
        except:
            agent_role = AgentRole.ATTACKER.value  # Default to attacker if conversion fails
        
        # If role doesn't exist in weights, use default weights
        if agent_role not in self.role_option_weights:
            return 1.0
            
        # Return the weight for this option
        return self.role_option_weights[agent_role].get(option_name, 1.0)
    
    def _check_option_conditions(self, agent_role, option_name, state):
        """Check if this option should be available based on role and state conditions"""
        # Convert agent_role to int using our safe method
        try:
            agent_role = int(self._safe_scalar(agent_role))
        except:
            agent_role = AgentRole.ATTACKER.value  # Default to attacker if conversion fails
            
        # Default conditions for common options
        if option_name == 'capture_flag':
            # Don't capture if already have flag
            if hasattr(state, 'agent_flags') and len(state.agent_flags) > 0:
                try:
                    agent_has_flag = self._safe_scalar(state.agent_flags[0])
                    if agent_has_flag > 0:
                        return False
                except:
                    pass  # On error, proceed with default behavior
                    
        elif option_name == 'return_to_base':
            # Only return to base if have flag or badly damaged
            if hasattr(state, 'agent_flags') and len(state.agent_flags) > 0:
                try:
                    agent_has_flag = self._safe_scalar(state.agent_flags[0])
                    if agent_has_flag <= 0:
                        # Check if agent is damaged
                        if hasattr(state, 'agent_health') and len(state.agent_health) > 0:
                            health = self._safe_scalar(state.agent_health[0])
                            if health > 0.4:  # Not badly damaged
                                return False
                except:
                    pass  # On error, proceed with default behavior
            
        elif option_name == 'defend_base':
            # Only defenders should defend base unless special circumstances
            if agent_role != AgentRole.DEFENDER.value:
                # Non-defenders can defend if flag is under high threat
                if hasattr(state, 'our_flag_threat'):
                    try:
                        flag_threat = self._safe_scalar(state.our_flag_threat)
                        if flag_threat < 0.7:  # Not high threat
                            return False
                    except:
                        pass  # On error, proceed with default behavior
        
        # All conditions passed
        return True

    def _safe_forward(self, input_tensor):
        """
        Safely perform forward pass through the network, handling shape mismatches.
        
        Args:
            input_tensor: Input tensor for the network
            
        Returns:
            Network output or None if error occurs
        """
        try:
            with torch.no_grad():
                if input_tensor.shape[1] != self.state_size:
                    if self.debug_level >= 1:
                        print(f"Input tensor shape mismatch: expected {self.state_size}, got {input_tensor.shape[1]}")
                    
                    # Pad or truncate the tensor to match expected size
                    if input_tensor.shape[1] < self.state_size:
                        # Pad with zeros
                        padding = torch.zeros(input_tensor.shape[0], self.state_size - input_tensor.shape[1])
                        input_tensor = torch.cat([input_tensor, padding], dim=1)
                    else:
                        # Truncate
                        input_tensor = input_tensor[:, :self.state_size]
                        
                return self.network(input_tensor)
        except Exception as e:
            if self.debug_level >= 1:
                print(f"Error in network forward pass: {e}")
            return None 

    def _safe_tensor_creation(self, features):
        """
        Safely create a tensor from a list of features, handling bad values.
        
        Args:
            features: List of feature values
            
        Returns:
            PyTorch tensor
        """
        try:
            # Replace any invalid values with 0
            clean_features = []
            for val in features:
                if isinstance(val, (int, float)) and not np.isnan(val) and not np.isinf(val):
                    clean_features.append(float(val))
                else:
                    clean_features.append(0.0)
                    
            return torch.tensor(clean_features, dtype=torch.float32).unsqueeze(0)
        except Exception as e:
            if self.debug_level >= 1:
                print(f"Error creating tensor: {e}")
            # Return a zero tensor of the expected size
            return torch.zeros(1, self.state_size)

    def _safe_scalar(self, value):
        """
        Safely convert a numpy array, tensor, or other value to a Python scalar.
        
        Args:
            value: Value to convert
            
        Returns:
            Python scalar value
        """
        try:
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    return float(value.item())
                else:
                    # If array has multiple values, return the first one
                    return float(value.flatten()[0])
            elif isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    return float(value.item())
                else:
                    return float(value.view(-1)[0].item())
            elif isinstance(value, (int, float)):
                return float(value)
            else:
                return float(value)
        except Exception as e:
            if self.debug_level >= 2:
                print(f"Error in safe_scalar: {e}")
            return 0.0 