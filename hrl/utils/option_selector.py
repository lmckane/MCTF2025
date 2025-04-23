import numpy as np
from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from hrl.utils.state_processor import ProcessedState
from hrl.policies.hierarchical_policy import Experience
from hrl.utils.team_coordinator import AgentRole

class OptionSelector:
    """Selects options based on state with team coordination awareness."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the option selector."""
        self.config = config
        # Extended state size to include role and coordination data
        self.state_size = 16  # Position[2], velocity[2], flags[1], tags[1], health[1], team[1], role[1], recommended_target[2], flag_threats[3], enemy_distance[1], team_spread[1]
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
                
            # Get agent role (default to ATTACKER if not available)
            agent_role = int(state.agent_roles[0]) if hasattr(state, 'agent_roles') and len(state.agent_roles) > 0 else AgentRole.ATTACKER.value
            
            # Get distance to recommended target
            has_recommended_target = hasattr(state, 'recommended_target') and len(state.recommended_target) == 2
            if has_recommended_target:
                target_distance = np.linalg.norm(state.agent_positions[0] - state.recommended_target)
            else:
                target_distance = 0.0
                
            # Get team spread - measure how spread out our team is
            team_spread = 0.0
            team_agents = []
            if len(state.agent_positions) > 1:
                for i in range(len(state.agent_positions)):
                    if i > 0 and state.agent_teams[i] == state.agent_teams[0]:  # Same team as our agent
                        team_agents.append(state.agent_positions[i])
                
                if team_agents:
                    distances = []
                    for pos in team_agents:
                        distances.append(np.linalg.norm(state.agent_positions[0] - pos))
                    team_spread = np.mean(distances) if distances else 0.0
            
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
            state_tensor = torch.tensor([
                # Basic agent features
                state.agent_positions[0][0], state.agent_positions[0][1],
                state.agent_velocities[0][0], state.agent_velocities[0][1],
                float(state.agent_flags[0]),
                float(state.agent_tags[0]),
                state.agent_health[0],
                float(state.agent_teams[0]),
                # Role and coordination features
                float(agent_role),
                target_distance,
                state.recommended_target[0] if has_recommended_target else 0.0,
                state.recommended_target[1] if has_recommended_target else 0.0,
                float(state.our_flag_threat) if hasattr(state, 'our_flag_threat') else 0.0,
                float(state.our_flag_captured) if hasattr(state, 'our_flag_captured') else 0.0,
                float(state.enemy_flag_captured) if hasattr(state, 'enemy_flag_captured') else 0.0,
                team_spread
            ], dtype=torch.float32).unsqueeze(0)
            
            # Get option scores from network
            option_scores = self.network(state_tensor)
            
            # Apply role-specific weights if available
            if agent_role in self.role_option_weights:
                role_weights = torch.tensor([self.role_option_weights[agent_role][option] 
                                          for option in self.config['options']])
                option_scores = option_scores * role_weights.unsqueeze(0)
            
            # Apply general option weights
            option_weights_tensor = torch.tensor([self.option_weights[option] for option in self.config['options']])
            weighted_scores = option_scores * option_weights_tensor.unsqueeze(0)
            
            # Apply counter-strategy adjustments based on opponent modeling
            if counter_strategy == "evasive":
                # Increase priority for return_to_base and decrease for aggressive options
                if position_danger > 0.5:  # High danger area
                    return_option_idx = self.config['options'].index('return_to_base') if 'return_to_base' in self.config['options'] else None
                    if return_option_idx is not None:
                        weighted_scores[0, return_option_idx] *= 2.0
                        
                    # Reduce aggressive options
                    for option in ['tag_enemy', 'attack_enemy']:
                        if option in self.config['options']:
                            option_idx = self.config['options'].index(option)
                            weighted_scores[0, option_idx] *= 0.5
                            
            elif counter_strategy == "flanking":
                # When counter-strategy is flanking, prioritize capture_flag and maneuverability
                capture_option_idx = self.config['options'].index('capture_flag') if 'capture_flag' in self.config['options'] else None
                if capture_option_idx is not None:
                    weighted_scores[0, capture_option_idx] *= 1.5
                    
            elif counter_strategy == "intercept":
                # Prioritize interception and tag_enemy
                intercept_option_idx = self.config['options'].index('intercept') if 'intercept' in self.config['options'] else None
                tag_option_idx = self.config['options'].index('tag_enemy') if 'tag_enemy' in self.config['options'] else None
                
                if intercept_option_idx is not None:
                    weighted_scores[0, intercept_option_idx] *= 1.7
                if tag_option_idx is not None:
                    weighted_scores[0, tag_option_idx] *= 1.3
                    
            elif counter_strategy == "defensive":
                # Prioritize defending base and territory control
                defend_option_idx = self.config['options'].index('defend_base') if 'defend_base' in self.config['options'] else None
                
                if defend_option_idx is not None:
                    weighted_scores[0, defend_option_idx] *= 1.5
                    
            elif counter_strategy == "disrupt":
                # Disrupt coordinated teams by focusing on tagging and flag possession
                tag_option_idx = self.config['options'].index('tag_enemy') if 'tag_enemy' in self.config['options'] else None
                capture_option_idx = self.config['options'].index('capture_flag') if 'capture_flag' in self.config['options'] else None
                
                if tag_option_idx is not None:
                    weighted_scores[0, tag_option_idx] *= 1.4
                if capture_option_idx is not None:
                    weighted_scores[0, capture_option_idx] *= 1.3
            
            # Strategic decision enhancements based on game state
            
            # 1. Flag carrier strategy
            if hasattr(state, 'agent_flags') and state.agent_flags[0] > 0:
                # If agent has flag, prioritize returning to base and evasion
                return_option_idx = self.config['options'].index('return_to_base') if 'return_to_base' in self.config['options'] else None
                
                # The closer to enemy, the more we prioritize returning to base
                if return_option_idx is not None:
                    enemy_proximity_factor = max(1.0, 2.0 / (enemy_distance + 0.1))  # Higher when enemies are close
                    weighted_scores[0, return_option_idx] *= 2.0 * enemy_proximity_factor
                
                # Deprioritize aggressive options when carrying flag
                for option in ['attack_enemy', 'tag_enemy', 'capture_flag']:
                    if option in self.config['options']:
                        option_idx = self.config['options'].index(option)
                        weighted_scores[0, option_idx] *= 0.3  # Greatly reduce aggressive behavior
            
            # 2. Role-specific strategic adjustments
            elif agent_role == AgentRole.DEFENDER.value:
                # Check if our flag is being threatened
                flag_under_threat = hasattr(state, 'our_flag_threat') and state.our_flag_threat > 0.5
                flag_captured = hasattr(state, 'our_flag_captured') and state.our_flag_captured
                
                # Defenders prioritize different options based on threat levels
                defend_option_idx = self.config['options'].index('defend_base') if 'defend_base' in self.config['options'] else None
                intercept_option_idx = self.config['options'].index('intercept') if 'intercept' in self.config['options'] else None
                
                if flag_captured:
                    # Flag is captured - prioritize intercepting enemy carrier
                    if intercept_option_idx is not None:
                        weighted_scores[0, intercept_option_idx] *= 3.0  # Strongly boost intercept
                elif flag_under_threat:
                    # Flag is under threat - switch between defending and intercepting
                    if defend_option_idx is not None:
                        weighted_scores[0, defend_option_idx] *= 2.0  # Boost defense
                    if intercept_option_idx is not None:
                        weighted_scores[0, intercept_option_idx] *= 1.5  # Moderately boost intercept
                else:
                    # No immediate threat - patrol and defend
                    if defend_option_idx is not None:
                        weighted_scores[0, defend_option_idx] *= 1.5  # Moderate defense priority
            
            elif agent_role == AgentRole.INTERCEPTOR.value:
                # Check for enemy threats
                flag_captured = hasattr(state, 'our_flag_captured') and state.our_flag_captured
                
                # Interceptors prioritize intercept, tag_enemy and strategic positioning
                intercept_option_idx = self.config['options'].index('intercept') if 'intercept' in self.config['options'] else None
                tag_option_idx = self.config['options'].index('tag_enemy') if 'tag_enemy' in self.config['options'] else None
                
                if flag_captured:
                    # Enemy has our flag - maximum priority on interception
                    if intercept_option_idx is not None:
                        weighted_scores[0, intercept_option_idx] *= 4.0  # Critical priority
                elif enemy_distance < 0.3:  # Enemy is close
                    # Close enemy - prioritize tagging
                    if tag_option_idx is not None:
                        weighted_scores[0, tag_option_idx] *= 2.5  # High priority on tagging
                else:
                    # No immediate threats - balance between interception and tagging
                    if intercept_option_idx is not None:
                        weighted_scores[0, intercept_option_idx] *= 1.5
                    if tag_option_idx is not None:
                        weighted_scores[0, tag_option_idx] *= 1.5
                        
                # Strategic consideration: if many enemies are already tagged, focus on other priorities
                if total_enemies > 0 and tagged_enemies / total_enemies > 0.5:
                    # Most enemies are tagged, consider helping with attack or defense
                    capture_option_idx = self.config['options'].index('capture_flag') if 'capture_flag' in self.config['options'] else None
                    if capture_option_idx is not None:
                        weighted_scores[0, capture_option_idx] *= 1.3  # Moderately boost attacking
            
            elif agent_role == AgentRole.ATTACKER.value:
                # Attackers focus on flag capture but with strategic considerations
                enemy_flag_captured = hasattr(state, 'enemy_flag_captured') and state.enemy_flag_captured
                
                # Different strategies based on flag status
                capture_option_idx = self.config['options'].index('capture_flag') if 'capture_flag' in self.config['options'] else None
                tag_option_idx = self.config['options'].index('tag_enemy') if 'tag_enemy' in self.config['options'] else None
                
                if enemy_flag_captured:
                    # Another team member has the flag - provide tactical support
                    if tag_option_idx is not None:
                        weighted_scores[0, tag_option_idx] *= 1.8  # Prioritize clearing path for flag carrier
                else:
                    # Flag not captured yet - focus on capturing with situational awareness
                    if capture_option_idx is not None:
                        # Scale capture priority based on proximity to flag (closer = higher priority)
                        proximity_factor = 1.0 + max(0, 1.0 - enemy_flag_distance) * 2.0
                        weighted_scores[0, capture_option_idx] *= 1.8 * proximity_factor
                    
                    # If enemy is very close, consider tagging first before resuming capture
                    if enemy_distance < 0.15 and tag_option_idx is not None:
                        weighted_scores[0, tag_option_idx] *= 1.5  # Moderately boost tagging when enemy very close
            
            # 3. Team formation considerations
            team_roles_count = {AgentRole.ATTACKER.value: 0, AgentRole.DEFENDER.value: 0, AgentRole.INTERCEPTOR.value: 0}
            if hasattr(state, 'agent_roles'):
                for i in range(len(state.agent_roles)):
                    if i > 0 and state.agent_teams[i] == state.agent_teams[0]:  # Same team
                        role = int(state.agent_roles[i])
                        if role in team_roles_count:
                            team_roles_count[role] += 1
            
            # Encourage diversification if team is too concentrated in one role
            max_role_count = max(team_roles_count.values()) if team_roles_count else 0
            if max_role_count >= 2 and agent_role in team_roles_count and team_roles_count[agent_role] == max_role_count:
                # This agent is in an overrepresented role - consider alternative options
                if agent_role == AgentRole.ATTACKER.value:
                    # Too many attackers, consider defensive play if needed
                    if hasattr(state, 'our_flag_threat') and state.our_flag_threat > 0.3:
                        defend_option_idx = self.config['options'].index('defend_base') if 'defend_base' in self.config['options'] else None
                        if defend_option_idx is not None:
                            weighted_scores[0, defend_option_idx] *= 1.2  # Slightly boost defense
                
                elif agent_role == AgentRole.DEFENDER.value:
                    # Too many defenders, consider attack if safe
                    if not (hasattr(state, 'our_flag_threat') and state.our_flag_threat > 0.3):
                        capture_option_idx = self.config['options'].index('capture_flag') if 'capture_flag' in self.config['options'] else None
                        if capture_option_idx is not None:
                            weighted_scores[0, capture_option_idx] *= 1.2  # Slightly boost attack
            
            # Print debug info occasionally
            if self.debug_level >= 2 and np.random.random() < 0.005:  # 0.5% chance to print debug info
                print("\nEnhanced Option Selection Debug:")
                print(f"  Agent role: {AgentRole(agent_role).name if agent_role in [r.value for r in AgentRole] else 'UNKNOWN'}")
                print(f"  Flag status: has_flag={state.agent_flags[0]}, our_flag_captured={state.our_flag_captured if hasattr(state, 'our_flag_captured') else 'N/A'}")
                print(f"  Threat level: {state.our_flag_threat if hasattr(state, 'our_flag_threat') else 'N/A'}")
                print(f"  Counter strategy: {counter_strategy}, Position danger: {position_danger:.2f}")
                print(f"  Distances: enemy={enemy_distance:.2f}, enemy_flag={enemy_flag_distance:.2f}, our_flag={our_flag_distance:.2f}")
                print("  Option scores:")
                for i, option in enumerate(self.config['options']):
                    role_weight = self.role_option_weights[agent_role][option] if agent_role in self.role_option_weights else 1.0
                    print(f"    {option}: raw={option_scores[0][i].item():.2f}, role_weight={role_weight:.2f}, "
                          f"general_weight={self.option_weights[option]:.2f}, final={weighted_scores[0][i].item():.2f}")
            
            # Select option with highest score
            option_idx = torch.argmax(weighted_scores).item()
            selected_option = self.config['options'][option_idx]
            
            return selected_option
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
        # Extract relevant features
        features = []
        
        # Add agent features
        features.extend(state.agent_positions.flatten())  # Shape: (num_agents * 2,)
        features.extend(state.agent_velocities.flatten())  # Shape: (num_agents * 2,)
        features.extend(state.agent_flags)  # Shape: (num_agents,)
        features.extend(state.agent_tags)  # Shape: (num_agents,)
        features.extend(state.agent_teams)  # Shape: (num_agents,)
        features.extend(state.agent_health)  # Shape: (num_agents,)
        
        # Add coordination features if available
        if hasattr(state, 'agent_roles'):
            features.extend(state.agent_roles)  # Shape: (num_agents,)
        
        if hasattr(state, 'recommended_target'):
            features.extend(state.recommended_target)  # Shape: (2,)
            
        if hasattr(state, 'our_flag_threat'):
            features.append(state.our_flag_threat)  # Shape: (1,)
            
        if hasattr(state, 'our_flag_captured'):
            features.append(float(state.our_flag_captured))  # Shape: (1,)
            
        if hasattr(state, 'enemy_flag_captured'):
            features.append(float(state.enemy_flag_captured))  # Shape: (1,)
        
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