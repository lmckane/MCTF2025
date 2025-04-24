import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum
import random


class AgentRole(Enum):
    """Possible roles for agents in the team."""
    ATTACKER = 0  # Focus on capturing enemy flags
    DEFENDER = 1  # Focus on defending own flag/territory
    INTERCEPTOR = 2  # Focus on tagging enemy agents, especially flag carriers


class TeamCoordinator:
    """
    Manages coordination between agents on the same team.
    
    Responsibilities:
    - Assign roles to agents based on team needs
    - Track enemy positions and movements
    - Coordinate defensive and offensive strategies
    - Provide additional state information to agents
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the team coordinator.
        
        Args:
            config: Configuration dictionary with parameters:
                - num_agents: Number of agents per team
                - role_update_frequency: How often to update roles (steps)
                - team_id: Which team this coordinator is for (default 0)
        """
        self.config = config
        self.num_agents = config.get('num_agents', 3)
        self.role_update_frequency = config.get('role_update_frequency', 10)
        self.team_id = config.get('team_id', 0)
        
        # Initialize agent roles
        self.agent_roles = {}
        self.step_counter = 0
        
        # Memory of previous states to track movement patterns
        self.state_history = []
        self.max_history_length = 5
        
        # Initialize threat assessment
        self.enemy_threat_levels = {}
        self.own_flag_threat = 0.0
        
    def reset(self):
        """Reset the coordinator state for a new episode."""
        self.agent_roles = {}
        self.step_counter = 0
        self.state_history = []
        self.enemy_threat_levels = {}
        self.own_flag_threat = 0.0
        
    def assign_initial_roles(self, state: Dict[str, Any]):
        """
        Assign initial roles to team agents.
        
        Args:
            state: Current game state
        """
        # Get agents on our team
        team_agents = [agent for agent in state['agents'] 
                     if agent['team'] == self.team_id]
        
        # More aggressive strategy: 2 attackers, 1 defender
        num_defenders = 1
        num_attackers = len(team_agents) - num_defenders
        
        # Ensure at least one attacker if possible
        if num_attackers < 1 and len(team_agents) > 0:
            num_defenders = len(team_agents) - 1
            num_attackers = 1
        
        # Create role assignments
        roles_to_assign = ([AgentRole.DEFENDER] * num_defenders + 
                         [AgentRole.ATTACKER] * num_attackers)
        
        # Shuffle roles to prevent predictable assignments
        random.shuffle(roles_to_assign)
        
        # Assign roles to agents
        self.agent_roles = {}
        for i, agent in enumerate(team_agents):
            # Add 'id' field if it doesn't exist
            if isinstance(agent, dict) and 'id' not in agent:
                # Use index as fallback ID
                agent['id'] = i
                
            agent_id = agent['id'] if isinstance(agent, dict) else agent.id
            self.agent_roles[agent_id] = roles_to_assign[i]
        
    def update_roles(self, state: Dict[str, Any]):
        """
        Update agent roles based on the current game state.
        
        Args:
            state: Current game state
        """
        self.step_counter += 1
        
        # Only update roles periodically to maintain consistency
        if self.step_counter % self.role_update_frequency != 0:
            return
            
        # Store state in history
        self.state_history.append(state)
        if len(self.state_history) > self.max_history_length:
            self.state_history.pop(0)
            
        # Assess threats and flag status
        self._assess_threats(state)
        
        # Get agents on our team
        team_agents = [agent for agent in state['agents'] 
                       if agent['team'] == self.team_id]
        
        # If no roles assigned yet, make initial assignments
        if not self.agent_roles:
            self.assign_initial_roles(state)
            return
            
        # Analyze game state to determine optimal role distribution
        our_flag_captured = self._is_our_flag_captured(state)
        we_have_enemy_flag = self._we_have_enemy_flag(state)
        
        # Calculate game progression (0 to 1)
        max_steps = 1000  # Adjust based on your game configuration
        game_progress = min(1.0, self.step_counter / max_steps)
        
        # Count tagged and untagged agents
        tagged_agents = sum(1 for agent in team_agents if (agent['is_tagged'] if isinstance(agent, dict) else agent.is_tagged))
        active_agents = len(team_agents) - tagged_agents
        
        # Adjusted role distribution based on game state
        if our_flag_captured:
            # Prioritize interceptors to recover our flag
            # Use 2 interceptors if we have enough active agents, otherwise use all for interception
            if active_agents >= 2:
                roles_to_assign = [AgentRole.INTERCEPTOR] * 2 + [AgentRole.ATTACKER] * (len(team_agents) - 2)
            else:
                roles_to_assign = [AgentRole.INTERCEPTOR] * active_agents + [AgentRole.ATTACKER] * (len(team_agents) - active_agents)
        elif we_have_enemy_flag:
            # Protect flag carrier and optimize for scoring
            flag_carrier_id = None
            for i, agent in enumerate(team_agents):
                # Add 'id' field if it doesn't exist
                if isinstance(agent, dict) and 'id' not in agent:
                    agent['id'] = i
                
                agent_has_flag = agent['has_flag'] if isinstance(agent, dict) else agent.has_flag
                if agent_has_flag:
                    flag_carrier_id = agent['id'] if isinstance(agent, dict) else agent.id
                    break
            
            # More defenders in early game, more attackers in late game
            if game_progress < 0.7:
                # Early/mid game: protect carrier with defenders
                roles_to_assign = [AgentRole.DEFENDER] * 2 + [AgentRole.ATTACKER] * (len(team_agents) - 2)
            else:
                # Late game: push for victory with more attackers
                roles_to_assign = [AgentRole.DEFENDER] * 1 + [AgentRole.ATTACKER] * (len(team_agents) - 1)
        elif self.own_flag_threat > 0.7:  # High threat to our flag
            # Increase defense when our flag is threatened
            roles_to_assign = [AgentRole.DEFENDER] * 2 + [AgentRole.ATTACKER] * (len(team_agents) - 2)
        elif self.own_flag_threat > 0.4:  # Medium threat
            # Balanced team with one interceptor to neutralize approaching threats
            roles_to_assign = [AgentRole.DEFENDER, AgentRole.INTERCEPTOR] + [AgentRole.ATTACKER] * (len(team_agents) - 2)
        else:
            # No immediate threats - adapt based on game progress
            if game_progress < 0.3:
                # Early game: aggressive stance with more attackers
                roles_to_assign = [AgentRole.DEFENDER] + [AgentRole.ATTACKER] * (len(team_agents) - 1)
            elif game_progress > 0.7:
                # Late game: more balanced approach with emphasis on defense if score is favorable
                if self._get_score_difference(state) > 0:
                    # Winning: more defense to secure victory
                    roles_to_assign = [AgentRole.DEFENDER] * 2 + [AgentRole.ATTACKER] * (len(team_agents) - 2)
                else:
                    # Losing: go aggressive
                    roles_to_assign = [AgentRole.DEFENDER] + [AgentRole.ATTACKER] * (len(team_agents) - 1)
            else:
                # Mid game: standard balanced distribution
                roles_to_assign = [AgentRole.DEFENDER] + [AgentRole.ATTACKER] * (len(team_agents) - 1)
            
        # Ensure we have at least one role per agent
        while len(roles_to_assign) < len(team_agents):
            roles_to_assign.append(AgentRole.ATTACKER)
            
        # Match roles with best-suited agents
        self._assign_roles_to_best_agents(team_agents, roles_to_assign, state)
            
    def _assign_roles_to_best_agents(self, team_agents: List[Dict], roles_to_assign: List[AgentRole], 
                                    state: Dict[str, Any]):
        """
        Assign roles to the most suitable agents based on position and status.
        
        Args:
            team_agents: List of agents on our team
            roles_to_assign: List of roles to assign
            state: Current game state
        """
        # Get key locations
        flag_position = self._get_our_flag_position(state)
        enemy_flag_position = self._get_enemy_flag_position(state)
        
        # Calculate suitability scores for each agent for each role
        suitability_scores = {}
        
        for i, agent in enumerate(team_agents):
            # Add 'id' field if it doesn't exist
            if isinstance(agent, dict) and 'id' not in agent:
                agent['id'] = i
                
            agent_id = agent['id'] if isinstance(agent, dict) else agent.id
            position = np.array(agent['position'] if isinstance(agent, dict) else agent.position)
            
            # Skip tagged agents, keep their current role
            is_tagged = agent['is_tagged'] if isinstance(agent, dict) else agent.is_tagged
            if is_tagged:
                continue
                
            # Calculate suitability for each role
            defender_score = 1.0 / (1.0 + np.linalg.norm(position - flag_position))
            attacker_score = 1.0 / (1.0 + np.linalg.norm(position - enemy_flag_position))
            
            # For interceptor, consider proximity to enemy agents, especially flag carriers
            interceptor_score = self._calculate_interceptor_suitability(agent, state)
            
            suitability_scores[agent_id] = {
                AgentRole.DEFENDER: defender_score,
                AgentRole.ATTACKER: attacker_score,
                AgentRole.INTERCEPTOR: interceptor_score
            }
            
        # Assign roles to maximize overall suitability
        assigned_agents = set()
        new_roles = {}
        
        # First, handle flag carriers - they should be attackers if they have enemy flag
        for i, agent in enumerate(team_agents):
            # Add 'id' field if it doesn't exist
            if isinstance(agent, dict) and 'id' not in agent:
                agent['id'] = i
                
            agent_id = agent['id'] if isinstance(agent, dict) else agent.id
            has_flag = agent['has_flag'] if isinstance(agent, dict) else agent.has_flag
            
            if has_flag:
                new_roles[agent_id] = AgentRole.ATTACKER
                assigned_agents.add(agent_id)
                # Remove an attacker role from the list as we've assigned one
                if AgentRole.ATTACKER in roles_to_assign:
                    roles_to_assign.remove(AgentRole.ATTACKER)
        
        # Assign remaining roles to maximize suitability
        for role in roles_to_assign:
            if not team_agents or all(
                ((agent['id'] if 'id' in agent else i) if isinstance(agent, dict) else agent.id) in assigned_agents 
                for i, agent in enumerate(team_agents)
            ):
                break
                
            # Find best agent for this role
            best_agent_id = None
            best_score = -1
            
            for i, agent in enumerate(team_agents):
                # Add 'id' field if it doesn't exist
                if isinstance(agent, dict) and 'id' not in agent:
                    agent['id'] = i
                    
                agent_id = agent['id'] if isinstance(agent, dict) else agent.id
                if agent_id in assigned_agents:
                    continue
                    
                if agent_id in suitability_scores and role in suitability_scores[agent_id]:
                    score = suitability_scores[agent_id][role]
                    if score > best_score:
                        best_score = score
                        best_agent_id = agent_id
            
            if best_agent_id is not None:
                new_roles[best_agent_id] = role
                assigned_agents.add(best_agent_id)
        
        # Update agent roles with new assignments
        self.agent_roles.update(new_roles)
    
    def get_agent_role(self, agent_id: int) -> AgentRole:
        """
        Get the assigned role for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Assigned role for the agent, defaults to ATTACKER if not assigned
        """
        return self.agent_roles.get(agent_id, AgentRole.ATTACKER)
    
    def _assess_threats(self, state: Dict[str, Any]):
        """
        Assess threats to our flag and bases.
        
        Args:
            state: Current game state
        """
        enemy_agents = [agent for agent in state['agents'] 
                       if agent['team'] != self.team_id]
        
        our_flag_position = self._get_our_flag_position(state)
        our_base_position = state['team_bases'][self.team_id]
        
        # Assess threat to our flag
        flag_threats = []
        for i, agent in enumerate(enemy_agents):
            # Add 'id' field if it doesn't exist
            if isinstance(agent, dict) and 'id' not in agent:
                agent['id'] = i + len(state['agents']) // 2  # Offset to avoid ID conflicts with team agents
                
            position = np.array(agent['position'] if isinstance(agent, dict) else agent.position)
            distance_to_flag = np.linalg.norm(position - our_flag_position)
            
            # Calculate threat level based on distance (closer = higher threat)
            # Scale to 0-1 range
            map_size = np.array(state.get('map_size', [100, 100]))
            map_diagonal = np.linalg.norm(map_size)
            threat = max(0, 1.0 - distance_to_flag / (map_diagonal * 0.5))
            
            # Tagged enemies pose no threat
            is_tagged = agent['is_tagged'] if isinstance(agent, dict) else agent.is_tagged
            if is_tagged:
                threat = 0
                
            flag_threats.append(threat)
            
            # Store individual threat levels
            agent_id = agent['id'] if isinstance(agent, dict) else agent.id
            self.enemy_threat_levels[agent_id] = threat
        
        # Overall flag threat is the maximum individual threat
        self.own_flag_threat = max(flag_threats) if flag_threats else 0.0
    
    def _calculate_interceptor_suitability(self, agent: Dict, state: Dict[str, Any]) -> float:
        """
        Calculate how suitable an agent is for the interceptor role.
        
        Args:
            agent: The agent to evaluate
            state: Current game state
            
        Returns:
            Suitability score for interceptor role
        """
        position = np.array(agent['position'] if isinstance(agent, dict) else agent.position)
        
        # Find enemy flag carrier if it exists
        enemy_carrier = None
        carrier_id = None
        for flag in state['flags']:
            flag_team = flag['team'] if isinstance(flag, dict) else flag.team
            flag_captured = flag['is_captured'] if isinstance(flag, dict) else flag.is_captured
            
            if flag_team == self.team_id and flag_captured:
                carrier_id = flag['carrier_id'] if isinstance(flag, dict) else flag.carrier_id
                # Find the carrier in agents
                for enemy_agent in state['agents']:
                    # Add 'id' field if it doesn't exist
                    if isinstance(enemy_agent, dict) and 'id' not in enemy_agent:
                        enemy_agent['id'] = state['agents'].index(enemy_agent)
                        
                    enemy_id = enemy_agent['id'] if isinstance(enemy_agent, dict) else enemy_agent.id
                    if enemy_id == carrier_id:
                        enemy_carrier = enemy_agent
                        break
        
        if enemy_carrier:
            carrier_pos = np.array(enemy_carrier['position'] if isinstance(enemy_carrier, dict) else enemy_carrier.position)
            enemy_base = np.array(state['team_bases'][1 - self.team_id])
            
            # Get carrier velocity/direction if available in history
            carrier_velocity = np.zeros(2)
            if len(self.state_history) >= 2:
                previous_state = self.state_history[-2]
                for prev_enemy in previous_state['agents']:
                    prev_enemy_id = prev_enemy['id'] if isinstance(prev_enemy, dict) else prev_enemy.id
                    if prev_enemy_id == carrier_id:
                        prev_pos = np.array(prev_enemy['position'] if isinstance(prev_enemy, dict) else prev_enemy.position)
                        # Calculate velocity vector
                        carrier_velocity = carrier_pos - prev_pos
                        break
            
            # Predict where carrier is heading
            # Blend between current velocity and direction to base
            direction_to_base = enemy_base - carrier_pos
            if np.linalg.norm(direction_to_base) > 0:
                direction_to_base = direction_to_base / np.linalg.norm(direction_to_base)
            
            # If carrier has clear velocity, weight it higher, otherwise assume they're heading to base
            if np.linalg.norm(carrier_velocity) > 0.5:
                carrier_velocity = carrier_velocity / np.linalg.norm(carrier_velocity)
                predicted_direction = carrier_velocity * 0.7 + direction_to_base * 0.3
            else:
                predicted_direction = direction_to_base
            
            if np.linalg.norm(predicted_direction) > 0:
                predicted_direction = predicted_direction / np.linalg.norm(predicted_direction)
            
            # Calculate optimal interception point based on relative speeds and positions
            my_speed = 1.0  # Assuming normalized speed
            enemy_speed = 0.9  # Slightly disadvantage enemy for safer interception
            
            # Estimate direct distance and time for both agents
            distance_to_carrier = np.linalg.norm(carrier_pos - position)
            
            # Interception time calculation - simplified for efficiency
            # Try several potential interception points along carrier's predicted path
            best_intercept_score = 0
            best_intercept_distance = distance_to_carrier
            
            for t in range(5, 40, 5):  # Try interception points at 5, 10, 15... units ahead
                potential_carrier_pos = carrier_pos + predicted_direction * t * enemy_speed
                
                # Check if this position is on the map
                map_size = np.array(state.get('map_size', [100, 100]))
                if np.any(potential_carrier_pos < 0) or np.any(potential_carrier_pos > map_size):
                    continue
                
                distance_to_intercept = np.linalg.norm(potential_carrier_pos - position)
                my_time = distance_to_intercept / my_speed
                carrier_time = t
                
                # If we can get there before or at the same time as carrier, it's a good intercept point
                if my_time <= carrier_time + 1:  # Add small buffer for safety
                    intercept_score = 1.0 / (1.0 + my_time)
                    if intercept_score > best_intercept_score:
                        best_intercept_score = intercept_score
                        best_intercept_distance = distance_to_intercept
            
            # If we can intercept, return high score, otherwise score based on direct distance
            if best_intercept_score > 0:
                return 3.0 * best_intercept_score  # Higher priority than standard interception
            
            # Fallback to direct distance if no good interception found
            return 2.0 / (1.0 + best_intercept_distance * 0.5)  # Reduce distance penalty
        
        # Otherwise, look for closest enemy in our territory
        enemy_in_territory = []
        for enemy in state['agents']:
            # Add 'id' field if it doesn't exist
            if isinstance(enemy, dict) and 'id' not in enemy:
                enemy['id'] = state['agents'].index(enemy)
                
            enemy_team = enemy['team'] if isinstance(enemy, dict) else enemy.team
            enemy_tagged = enemy['is_tagged'] if isinstance(enemy, dict) else enemy.is_tagged
            
            if enemy_team != self.team_id and not enemy_tagged:
                enemy_pos = np.array(enemy['position'] if isinstance(enemy, dict) else enemy.position)
                # Check if enemy is in our territory
                if self._is_in_our_territory(enemy_pos, state):
                    # Calculate threat level based on proximity to our flag
                    our_flag_pos = self._get_our_flag_position(state)
                    distance_to_flag = np.linalg.norm(enemy_pos - our_flag_pos)
                    distance_to_agent = np.linalg.norm(position - enemy_pos)
                    
                    # Higher score for enemies closer to our flag
                    flag_threat = max(0, 1.0 - distance_to_flag / 100)
                    intercept_score = (1.0 / (1.0 + distance_to_agent)) * (1.0 + flag_threat * 2)
                    enemy_in_territory.append((distance_to_agent, intercept_score, enemy_pos))
        
        if enemy_in_territory:
            # Sort by intercept score (higher is better)
            enemy_in_territory.sort(key=lambda x: x[1], reverse=True)
            best_score = enemy_in_territory[0][1]
            return best_score
        
        # If no specific targets, lower suitability but still non-zero
        return 0.3
    
    def _is_in_our_territory(self, position: np.ndarray, state: Dict[str, Any]) -> bool:
        """
        Check if a position is in our territory.
        
        Args:
            position: Position to check
            state: Current game state
            
        Returns:
            True if position is in our territory, False otherwise
        """
        try:
            # Safe conversion to numpy array
            position = self._safe_array(position)
            
            # Get map parameters
            map_size = self._safe_array(state.get('map_size', [100, 100]))
            map_center = map_size / 2
            
            # Simple territory division - just use x-coordinate
            # Adjust based on team ID (team 0 is on left, team 1 is on right)
            if self.team_id == 0:
                return position[0] < map_center[0]
            else:
                return position[0] >= map_center[0]
        except Exception as e:
            print(f"Error in _is_in_our_territory: {e}")
            return False
    
    def _is_our_flag_captured(self, state: Dict[str, Any]) -> bool:
        """Check if our flag is captured."""
        for flag in state['flags']:
            flag_team = flag['team'] if isinstance(flag, dict) else flag.team
            flag_captured = flag['is_captured'] if isinstance(flag, dict) else flag.is_captured
            
            if flag_team == self.team_id and flag_captured:
                return True
        return False
    
    def _we_have_enemy_flag(self, state: Dict[str, Any]) -> bool:
        """Check if we have captured an enemy flag."""
        for agent in state['agents']:
            agent_team = agent['team'] if isinstance(agent, dict) else agent.team
            agent_has_flag = agent['has_flag'] if isinstance(agent, dict) else agent.has_flag
            
            if agent_team == self.team_id and agent_has_flag:
                return True
        return False
    
    def _get_our_flag_position(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get the position of our flag.
        
        Args:
            state: Current game state
            
        Returns:
            Position of our flag as numpy array
        """
        try:
            for flag in state['flags']:
                flag_team = flag['team'] if isinstance(flag, dict) else flag.team
                if flag_team == self.team_id:
                    position = flag['position'] if isinstance(flag, dict) else flag.position
                    return self._safe_array(position)
            # If flag not found, use our base position
            if 'team_bases' in state and self.team_id < len(state['team_bases']):
                return self._safe_array(state['team_bases'][self.team_id])
            return np.array([0, 0])  # Fallback
        except Exception as e:
            print(f"Error in _get_our_flag_position: {e}")
            return np.array([0, 0])  # Fallback
    
    def _get_enemy_flag_position(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Get the position of enemy flag.
        
        Args:
            state: Current game state
            
        Returns:
            Position of enemy flag as numpy array
        """
        try:
            for flag in state['flags']:
                flag_team = flag['team'] if isinstance(flag, dict) else flag.team
                if flag_team != self.team_id:
                    position = flag['position'] if isinstance(flag, dict) else flag.position
                    return self._safe_array(position)
            # If flag not found, use enemy base position
            if 'team_bases' in state and (1 - self.team_id) < len(state['team_bases']):
                return self._safe_array(state['team_bases'][1 - self.team_id])
            return np.array([100, 100])  # Fallback
        except Exception as e:
            print(f"Error in _get_enemy_flag_position: {e}")
            return np.array([100, 100])  # Fallback
    
    def get_coordination_data(self, agent_id: int, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get coordination data for a specific agent.
        
        Args:
            agent_id: ID of the agent requesting data
            state: Current game state
            
        Returns:
            Dictionary with coordination information
        """
        # First make sure roles are updated
        if not self.agent_roles:
            self.assign_initial_roles(state)
            
        agent_role = self.get_agent_role(agent_id)
        
        # Find all team agents
        team_agents = []
        for i, agent in enumerate(state['agents']):
            # Add 'id' field if it doesn't exist
            if isinstance(agent, dict) and 'id' not in agent:
                agent['id'] = i
                
            agent_team = agent['team'] if isinstance(agent, dict) else agent.team
            if agent_team == self.team_id:
                team_agents.append(agent)
                
        # Get positions of all team members
        team_positions = {}
        team_roles = {}
        for i, agent in enumerate(team_agents):
            # Add 'id' field if it doesn't exist
            if isinstance(agent, dict) and 'id' not in agent:
                agent['id'] = i
                
            a_id = agent['id'] if isinstance(agent, dict) else agent.id
            team_positions[a_id] = np.array(agent['position'] if isinstance(agent, dict) else agent.position)
            team_roles[a_id] = self.get_agent_role(a_id)
            
        # Determine action recommendations based on role
        recommended_target = self._get_recommended_target(agent_id, agent_role, state)
        
        # Flag and threat information
        our_flag_position = self._get_our_flag_position(state)
        enemy_flag_position = self._get_enemy_flag_position(state)
        our_flag_captured = self._is_our_flag_captured(state)
        enemy_flag_captured = any(
            (agent['has_flag'] if isinstance(agent, dict) else agent.has_flag)
            for agent in team_agents
        )
        
        return {
            'agent_role': agent_role,
            'team_positions': team_positions,
            'team_roles': team_roles,
            'recommended_target': recommended_target,
            'our_flag_position': our_flag_position,
            'enemy_flag_position': enemy_flag_position,
            'our_flag_captured': our_flag_captured,
            'enemy_flag_captured': enemy_flag_captured,
            'own_flag_threat': self.own_flag_threat,
            'enemy_threats': self.enemy_threat_levels
        }
    
    def _get_recommended_target(self, agent_id: int, role: AgentRole, state: Dict[str, Any]) -> np.ndarray:
        """
        Get recommended target position for an agent based on their role.
        
        Args:
            agent_id: ID of the agent
            role: Role of the agent
            state: Current game state
            
        Returns:
            Recommended target position as numpy array
        """
        # Find the agent
        agent = None
        for i, a in enumerate(state['agents']):
            # Add 'id' field if it doesn't exist
            if isinstance(a, dict) and 'id' not in a:
                a['id'] = i
                
            a_id = a['id'] if isinstance(a, dict) else a.id
            if a_id == agent_id:
                agent = a
                break
                
        if agent is None:
            # Fallback if agent not found
            return np.array([0, 0])
            
        agent_position = np.array(agent['position'] if isinstance(agent, dict) else agent.position)
        agent_has_flag = agent['has_flag'] if isinstance(agent, dict) else agent.has_flag
        agent_is_tagged = agent['is_tagged'] if isinstance(agent, dict) else agent.is_tagged
        
        # Get map parameters
        map_size = np.array(state.get('map_size', [100, 100]))
        map_center = map_size / 2
        
        # Get key positions
        our_base = np.array(state['team_bases'][self.team_id])
        enemy_base = np.array(state['team_bases'][1 - self.team_id])
        
        # If agent is tagged, return to base for recovery
        if agent_is_tagged:
            return our_base
        
        # If agent has flag, calculate optimal path to base considering enemy positions
        if agent_has_flag:
            # Get enemy positions to determine safe path
            enemy_positions = []
            for enemy in state['agents']:
                enemy_team = enemy['team'] if isinstance(enemy, dict) else enemy.team
                enemy_tagged = enemy['is_tagged'] if isinstance(enemy, dict) else enemy.is_tagged
                if enemy_team != self.team_id and not enemy_tagged:
                    enemy_positions.append(np.array(enemy['position'] if isinstance(enemy, dict) else enemy.position))
            
            if enemy_positions:
                # Calculate direct path
                direct_path = our_base - agent_position
                direct_distance = np.linalg.norm(direct_path)
                
                # Check if enemies are in the direct path
                path_is_safe = True
                for enemy_pos in enemy_positions:
                    # Project enemy position onto path
                    if direct_distance > 0:
                        unit_path = direct_path / direct_distance
                        enemy_projection = np.dot(enemy_pos - agent_position, unit_path)
                        
                        if 0 < enemy_projection < direct_distance:
                            # Calculate perpendicular distance from enemy to path
                            perp_vector = enemy_pos - (agent_position + unit_path * enemy_projection)
                            perp_distance = np.linalg.norm(perp_vector)
                            
                            # If enemy is close to path, mark as unsafe
                            if perp_distance < 20:  # Safety radius
                                path_is_safe = False
                                break
                
                if not path_is_safe:
                    # Try alternate paths - flank around perceived enemies
                    # Use perpendicular vector to direct path for flanking
                    perp_vector = np.array([-direct_path[1], direct_path[0]])
                    if np.linalg.norm(perp_vector) > 0:
                        perp_vector = self._safe_normalize(perp_vector)
                        
                        # Check both left and right flanks for safety
                        left_flank = agent_position + perp_vector * 30  # 30 units to the left
                        right_flank = agent_position - perp_vector * 30  # 30 units to the right
                        
                        # Choose flank that's more distant from enemies
                        left_safety = min(np.linalg.norm(enemy_pos - left_flank) for enemy_pos in enemy_positions)
                        right_safety = min(np.linalg.norm(enemy_pos - right_flank) for enemy_pos in enemy_positions)
                        
                        if left_safety > right_safety:
                            # Left flank is safer
                            waypoint = left_flank
                        else:
                            # Right flank is safer
                            waypoint = right_flank
                        
                        # Ensure waypoint is within map bounds (with safety margin)
                        map_margin = map_size * 0.1
                        # Create lower and upper bounds as separate arrays
                        lower_bound = np.array([map_margin, map_margin])
                        upper_bound = map_size - map_margin
                        waypoint = self._safe_clip(waypoint, lower_bound, upper_bound)
                        
                        # Return the point
                        return waypoint
            
            # Default: return directly to base if path is safe or no better path found
            return our_base
            
        # Role-specific targets
        if role == AgentRole.DEFENDER:
            # Defend our flag/territory with improved positioning
            our_flag_captured = self._is_our_flag_captured(state)
            
            if our_flag_captured:
                # Find flag carrier
                carrier_pos = None
                for flag in state['flags']:
                    flag_team = flag['team'] if isinstance(flag, dict) else flag.team
                    flag_captured = flag['is_captured'] if isinstance(flag, dict) else flag.is_captured
                    
                    if flag_team == self.team_id and flag_captured:
                        carrier_id = flag['carrier_id'] if isinstance(flag, dict) else flag.carrier_id
                        # Find the carrier
                        for enemy in state['agents']:
                            enemy_id = enemy['id'] if isinstance(enemy, dict) else enemy.id
                            if enemy_id == carrier_id:
                                carrier_pos = np.array(enemy['position'] if isinstance(enemy, dict) else enemy.position)
                                break
                
                if carrier_pos is not None:
                    # Intercept along path to enemy base
                    direction_to_enemy_base = enemy_base - carrier_pos
                    if np.linalg.norm(direction_to_enemy_base) > 0:
                        intercept_direction = direction_to_enemy_base / np.linalg.norm(direction_to_enemy_base)
                        # Get ahead of carrier toward their destination
                        return carrier_pos + intercept_direction * 20
                else:
                    # If carrier not found but flag is captured, guard path between enemy base and our base
                    midpoint = (our_base + enemy_base) / 2
                    return midpoint
            else:
                # Flag is safe - implement improved defensive positioning
                flag_pos = self._get_our_flag_position(state)
                
                # Identify threat directions based on enemy positions
                threat_vector = np.zeros(2)
                threat_count = 0
                
                for enemy in state['agents']:
                    enemy_team = enemy['team'] if isinstance(enemy, dict) else enemy.team
                    if enemy_team != self.team_id:
                        enemy_pos = np.array(enemy['position'] if isinstance(enemy, dict) else enemy.position)
                        enemy_distance = np.linalg.norm(flag_pos - enemy_pos)
                        
                        # Consider enemies within a certain radius as threats
                        if enemy_distance < 50:  # Threat detection radius
                            enemy_direction = flag_pos - enemy_pos
                            if np.linalg.norm(enemy_direction) > 0:
                                # Weight by inverse distance (closer = higher threat)
                                direction_weight = 1.0 / max(10, enemy_distance)
                                threat_vector += enemy_direction * direction_weight
                                threat_count += 1
                
                if threat_count > 0 and np.linalg.norm(threat_vector) > 0:
                    # There are threats - position between flag and threat
                    threat_vector = threat_vector / np.linalg.norm(threat_vector)
                    
                    # If threats from multiple directions, position closer to flag
                    if threat_count > 1:
                        return flag_pos - threat_vector * 8  # Closer defensive position
                    else:
                        return flag_pos - threat_vector * 15  # Standard defensive position
                else:
                    # No immediate threats - patrol strategically around flag
                    # Position between flag and the direction of enemy base or map center
                    center_dir = map_center - flag_pos
                    enemy_dir = enemy_base - flag_pos
                    
                    # Weight between center and enemy directions
                    patrol_dir = center_dir * 0.3 + enemy_dir * 0.7
                    
                    if np.linalg.norm(patrol_dir) > 0:
                        patrol_dir = self._safe_normalize(patrol_dir)
                        return flag_pos + patrol_dir * 15  # Patrol 15 units away from flag
                    else:
                        return flag_pos  # Fallback to flag position
                
        elif role == AgentRole.ATTACKER:
            # Enhanced attacking strategy
            enemy_flag_pos = self._get_enemy_flag_position(state)
            enemy_flag_captured = any(
                (agent['has_flag'] if isinstance(agent, dict) else agent.has_flag)
                for agent in state['agents'] if agent['team'] == self.team_id
            )
            
            if enemy_flag_captured and not agent_has_flag:
                # Flag is captured by teammate - switch to supportive role
                # Find flag carrier teammate
                carrier = None
                for teammate in state['agents']:
                    tm_team = teammate['team'] if isinstance(teammate, dict) else teammate.team
                    tm_has_flag = teammate['has_flag'] if isinstance(teammate, dict) else teammate.has_flag
                    
                    if tm_team == self.team_id and tm_has_flag:
                        carrier = teammate
                        break
                
                if carrier:
                    carrier_pos = np.array(carrier['position'] if isinstance(carrier, dict) else carrier.position)
                    
                    # Check for nearby enemies that might threaten the carrier
                    nearby_enemies = []
                    for enemy in state['agents']:
                        enemy_team = enemy['team'] if isinstance(enemy, dict) else enemy.team
                        enemy_tagged = enemy['is_tagged'] if isinstance(enemy, dict) else enemy.is_tagged
                        
                        if enemy_team != self.team_id and not enemy_tagged:
                            enemy_pos = np.array(enemy['position'] if isinstance(enemy, dict) else enemy.position)
                            enemy_distance = np.linalg.norm(carrier_pos - enemy_pos)
                            
                            if enemy_distance < 40:  # Close to carrier
                                nearby_enemies.append((enemy_pos, enemy_distance))
                    
                    if nearby_enemies:
                        # Prioritize intercepting the closest enemy to carrier
                        nearby_enemies.sort(key=lambda x: x[1])
                        closest_enemy_pos = nearby_enemies[0][0]
                        
                        # Target the enemy
                        return closest_enemy_pos
                    else:
                        # No immediate threats - position between carrier and base
                        # to create a safe corridor
                        direction_to_base = our_base - carrier_pos
                        if np.linalg.norm(direction_to_base) > 0:
                            direction_to_base = direction_to_base / np.linalg.norm(direction_to_base)
                            # Position ahead of carrier toward base
                            escort_pos = carrier_pos + direction_to_base * 20
                            
                            # Offset to side to create wider corridor
                            perp_vector = np.array([-direction_to_base[1], direction_to_base[0]])
                            if agent_id % 2 == 0:  # Alternate sides for different agents
                                escort_pos += perp_vector * 15
                            else:
                                escort_pos -= perp_vector * 15
                                
                            return escort_pos
                
                # Default to guarding midway to our base
                return (our_base + enemy_flag_pos) / 2
            
            # Attack enemy flag with improved awareness
            # Check if the direct path is too dangerous
            enemy_positions = []
            for enemy in state['agents']:
                enemy_team = enemy['team'] if isinstance(enemy, dict) else enemy.team
                enemy_tagged = enemy['is_tagged'] if isinstance(enemy, dict) else enemy.is_tagged
                
                if enemy_team != self.team_id and not enemy_tagged:
                    enemy_positions.append(np.array(enemy['position'] if isinstance(enemy, dict) else enemy.position))
            
            # Check if a direct approach to flag is dangerous
            direct_path_dangerous = False
            direct_vector = enemy_flag_pos - agent_position
            direct_distance = np.linalg.norm(direct_vector)
            
            if enemy_positions and direct_distance > 0:
                unit_vector = direct_vector / direct_distance
                
                for enemy_pos in enemy_positions:
                    enemy_to_agent = agent_position - enemy_pos
                    enemy_to_flag = enemy_flag_pos - enemy_pos
                    
                    # Calculate if enemy is between agent and flag (or close to the path)
                    agent_enemy_dist = np.linalg.norm(enemy_to_agent)
                    if agent_enemy_dist < direct_distance:
                        # Project enemy position onto direct path
                        projection = np.dot(enemy_to_agent, unit_vector)
                        
                        if -10 < projection < direct_distance + 10:
                            # Calculate perpendicular distance from enemy to path
                            perp_vector = enemy_to_agent - unit_vector * projection
                            perp_distance = np.linalg.norm(perp_vector)
                            
                            # Enemy is close to direct path
                            if perp_distance < 15:
                                direct_path_dangerous = True
                                break
            
            if direct_path_dangerous:
                # Calculate alternate approach to flag - flanking maneuver
                # Get perpendicular vectors to the direct path
                if np.linalg.norm(direct_vector) > 0:
                    perp_vector = np.array([-direct_vector[1], direct_vector[0]])
                    perp_vector = self._safe_normalize(perp_vector)
                    
                    # Enhanced flanking approach
                    # Determine multiple potential flanking paths
                    flank_angles = [30, 45, 60]  # Angles in degrees
                    flank_distances = [25, 35, 45]  # Distances to try
                    
                    potential_flanks = []
                    for angle_deg in flank_angles:
                        for distance in flank_distances:
                            angle_rad = angle_deg * np.pi / 180.0
                            
                            # Calculate left and right flanking points using rotation matrices
                            cos_angle = np.cos(angle_rad)
                            sin_angle = np.sin(angle_rad)
                            
                            # Left flank (rotate counter-clockwise)
                            left_vector = np.array([
                                direct_vector[0] * cos_angle - direct_vector[1] * sin_angle,
                                direct_vector[0] * sin_angle + direct_vector[1] * cos_angle
                            ])
                            left_vector = left_vector / np.linalg.norm(left_vector)
                            left_flank = agent_position + left_vector * distance
                            
                            # Right flank (rotate clockwise)
                            right_vector = np.array([
                                direct_vector[0] * cos_angle + direct_vector[1] * sin_angle,
                                -direct_vector[0] * sin_angle + direct_vector[1] * cos_angle
                            ])
                            right_vector = right_vector / np.linalg.norm(right_vector)
                            right_flank = agent_position + right_vector * distance
                            
                            # Calculate safety scores
                            left_safety = self._calculate_path_safety(agent_position, left_flank, enemy_positions, state)
                            right_safety = self._calculate_path_safety(agent_position, right_flank, enemy_positions, state)
                            
                            # Consider distance to enemy flag from this point
                            left_flag_dist = np.linalg.norm(left_flank - enemy_flag_pos)
                            right_flag_dist = np.linalg.norm(right_flank - enemy_flag_pos)
                            
                            # Combine safety with progress toward goal
                            # Higher is better for both metrics
                            left_score = left_safety * 2.0 - left_flag_dist * 0.01
                            right_score = right_safety * 2.0 - right_flag_dist * 0.01
                            
                            # Ensure points are within map bounds
                            map_size = self._safe_array(state.get('map_size', [100, 100]))
                            try:
                                if self._safe_compare(left_flank, np.array([0, 0]), 'all_gte') and self._safe_compare(left_flank, map_size, 'all_lte'):
                                    potential_flanks.append((left_flank, left_score))
                                if self._safe_compare(right_flank, np.array([0, 0]), 'all_gte') and self._safe_compare(right_flank, map_size, 'all_lte'):
                                    potential_flanks.append((right_flank, right_score))
                            except Exception as e:
                                print(f"Error in flank bounds check: {e}")
                                # Fallback behavior
                                potential_flanks.append((left_flank, left_score))
                                potential_flanks.append((right_flank, right_score))
                    
                    # Choose the best flanking option
                    if potential_flanks:
                        potential_flanks.sort(key=lambda x: x[1], reverse=True)
                        best_flank, _ = potential_flanks[0]
                        
                        # Ensure waypoint is within map bounds (with safety margin)
                        map_margin = map_size * 0.1
                        # Create lower and upper bounds as separate arrays
                        lower_bound = np.array([map_margin, map_margin])
                        upper_bound = map_size - map_margin
                        flank_point = self._safe_clip(best_flank, lower_bound, upper_bound)
                        return flank_point
                    
                    # Fallback to original logic if no good flanks found
                    # Check both potential flanking paths
                    left_flank = agent_position + perp_vector * 30
                    right_flank = agent_position - perp_vector * 30
                    
                    # Choose safer flank based on enemy proximity
                    left_safety = min((np.linalg.norm(enemy_pos - left_flank) for enemy_pos in enemy_positions), default=float('inf'))
                    right_safety = min((np.linalg.norm(enemy_pos - right_flank) for enemy_pos in enemy_positions), default=float('inf'))
                    
                    # Choose the safer flank as waypoint
                    if left_safety > right_safety:
                        flank_point = left_flank
                    else:
                        flank_point = right_flank
                    
                    # Ensure waypoint is within map bounds (with safety margin)
                    map_margin = map_size * 0.1
                    # Create lower and upper bounds as separate arrays
                    lower_bound = np.array([map_margin, map_margin])
                    upper_bound = map_size - map_margin
                    flank_point = self._safe_clip(flank_point, lower_bound, upper_bound)
                    return flank_point
            
            # Default to direct approach to flag if no enemies blocking
            return enemy_flag_pos
            
        elif role == AgentRole.INTERCEPTOR:
            # Enhanced interceptor strategy with improved targeting
            # Priority 1: Enemy with our flag
            flag_carrier_pos = None
            for flag in state['flags']:
                flag_team = flag['team'] if isinstance(flag, dict) else flag.team
                flag_captured = flag['is_captured'] if isinstance(flag, dict) else flag.is_captured
                
                if flag_team == self.team_id and flag_captured:
                    carrier_id = flag['carrier_id'] if isinstance(flag, dict) else flag.carrier_id
                    for enemy in state['agents']:
                        enemy_id = enemy['id'] if isinstance(enemy, dict) else enemy.id
                        if enemy_id == carrier_id:
                            flag_carrier_pos = np.array(enemy['position'] if isinstance(enemy, dict) else enemy.position)
                            break
            
            if flag_carrier_pos is not None:
                # Calculate where carrier is likely heading (toward their base)
                carrier_to_base = enemy_base - flag_carrier_pos
                if np.linalg.norm(carrier_to_base) > 0:
                    carrier_direction = self._safe_normalize(carrier_to_base)
                    
                    # Intercept ahead of carrier's path
                    intercept_distance = min(30.0, np.linalg.norm(carrier_to_base) * 0.5)
                    intercept_point = flag_carrier_pos + carrier_direction * intercept_distance
                    
                    # Calculate if we should try to cut off carrier instead
                    our_direct_distance = np.linalg.norm(agent_position - enemy_base)
                    carrier_remaining_distance = np.linalg.norm(flag_carrier_pos - enemy_base)
                    
                    if our_direct_distance < carrier_remaining_distance * 0.8:
                        # We can potentially cut them off at their base
                        cutoff_point = enemy_base - carrier_direction * 15  # Just ahead of their base
                        return cutoff_point
                    
                    return intercept_point
                else:
                    return flag_carrier_pos
            
            # Priority 2: Enemies in our territory with improved targeting
            enemies_in_territory = []
            for enemy in state['agents']:
                enemy_team = enemy['team'] if isinstance(enemy, dict) else enemy.team
                enemy_tagged = enemy['is_tagged'] if isinstance(enemy, dict) else enemy.is_tagged
                
                if enemy_team != self.team_id and not enemy_tagged:
                    enemy_pos = np.array(enemy['position'] if isinstance(enemy, dict) else enemy.position)
                    if self._is_in_our_territory(enemy_pos, state):
                        # Calculate threat level based on proximity to our flag
                        our_flag_pos = self._get_our_flag_position(state)
                        distance_to_flag = np.linalg.norm(enemy_pos - our_flag_pos)
                        
                        # Higher threat if closer to our flag, scaled by distance
                        threat_level = 100.0 / max(10.0, distance_to_flag)
                        
                        # Distance from agent to enemy
                        dist_to_agent = np.linalg.norm(agent_position - enemy_pos)
                        
                        # Combine factors - prioritize high threats that are reasonably close
                        priority_score = threat_level * (100.0 / max(10.0, dist_to_agent))
                        
                        enemies_in_territory.append((enemy_pos, priority_score))
            
            if enemies_in_territory:
                # Target highest priority enemy
                enemies_in_territory.sort(key=lambda x: x[1], reverse=True)
                return enemies_in_territory[0][0]
            
            # Priority 3: Advanced patrol strategy
            our_flag_pos = self._get_our_flag_position(state)
            
            # Calculate threat direction based on enemy positions
            threat_direction = np.zeros(2)
            total_weight = 0
            
            for enemy in state['agents']:
                enemy_team = enemy['team'] if isinstance(enemy, dict) else enemy.team
                enemy_tagged = enemy['is_tagged'] if isinstance(enemy, dict) else enemy.is_tagged
                
                if enemy_team != self.team_id and not enemy_tagged:
                    enemy_pos = np.array(enemy['position'] if isinstance(enemy, dict) else enemy.position)
                    enemy_to_flag = our_flag_pos - enemy_pos
                    distance = np.linalg.norm(enemy_to_flag)
                    
                    # Weight inversely by distance
                    weight = 1.0 / max(15.0, distance)
                    threat_direction += enemy_to_flag * weight
                    total_weight += weight
            
            if total_weight > 0:
                # There is a perceived threat direction
                threat_direction = threat_direction / total_weight
                
                # Position in the direction of the threat
                patrol_point = our_flag_pos - threat_direction * 25
                
                # Make sure patrol point is within map bounds
                patrol_point = self._safe_clip(patrol_point, np.array([0, 0]), map_size)
                return patrol_point
            else:
                # No clear threats - patrol around flag with dynamic pattern
                angle = (self.step_counter / 10.0) % (2 * np.pi)
                patrol_radius = 25
                
                # Create elliptical patrol that puts more emphasis on direction of enemy base
                direction_to_enemy = enemy_base - our_flag_pos
                if np.linalg.norm(direction_to_enemy) > 0:
                    direction_to_enemy = self._safe_normalize(direction_to_enemy)
                    perpendicular = np.array([-direction_to_enemy[1], direction_to_enemy[0]])
                    
                    # Elongate patrol in the direction of potential threats
                    x_component = direction_to_enemy * patrol_radius * 1.5 * np.cos(angle)
                    y_component = perpendicular * patrol_radius * np.sin(angle)
                    
                    patrol_pos = our_flag_pos + x_component + y_component
                else:
                    # Standard circular patrol if no clear direction
                    patrol_pos = our_flag_pos + np.array([patrol_radius * np.cos(angle), patrol_radius * np.sin(angle)])
                
                # Make sure patrol point is within map bounds
                patrol_pos = self._safe_clip(patrol_pos, np.array([0, 0]), map_size)
                return patrol_pos
        
        # Default: return to base
        return our_base 

    def _get_score_difference(self, state: Dict[str, Any]) -> int:
        """
        Calculate the score difference between our team and opponent.
        
        Args:
            state: Current game state
            
        Returns:
            Score difference (positive if we're ahead, negative if behind)
        """
        if 'team_scores' in state:
            our_score = state['team_scores'][self.team_id]
            enemy_score = state['team_scores'][1 - self.team_id]
            return our_score - enemy_score
        return 0 

    def _calculate_path_safety(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                             enemy_positions: List[np.ndarray], state: Dict[str, Any]) -> float:
        """
        Calculate how safe a path is based on enemy positions.
        
        Args:
            start_pos: Starting position
            end_pos: Ending position
            enemy_positions: List of enemy positions
            state: Current game state
            
        Returns:
            Safety score (higher is safer)
        """
        # If no enemies, path is completely safe
        if not enemy_positions:
            return 100.0
            
        path_vector = end_pos - start_pos
        path_length = np.linalg.norm(path_vector)
        
        if path_length < 0.001:
            return 0.0  # Zero-length path
            
        path_direction = path_vector / path_length
        
        # Calculate minimum safe distance from any enemy
        safety_radius = 20.0
        
        # Discretize the path and check safety at each point
        num_checks = max(3, int(path_length / 10))
        min_safety = float('inf')
        
        for i in range(num_checks + 1):
            t = i / num_checks
            check_point = start_pos + t * path_vector
            
            # Calculate safety at this point
            min_enemy_dist = min(np.linalg.norm(check_point - enemy_pos) for enemy_pos in enemy_positions)
            
            # Update minimum safety along path
            min_safety = min(min_safety, min_enemy_dist)
        
        # Calculate safety score - higher is safer
        # Normalize to 0-100 range
        if min_safety >= safety_radius * 2:
            return 100.0  # Very safe
        elif min_safety <= safety_radius * 0.5:
            return 10.0   # Dangerous but not completely unsafe
        else:
            # Linear scaling between danger and safety thresholds
            normalized_safety = (min_safety - safety_radius * 0.5) / (safety_radius * 1.5)
            return 10.0 + 90.0 * normalized_safety 

    def _safe_clip(self, value, min_val, max_val):
        """
        Safely clips a value between minimum and maximum values,
        ensuring all inputs are proper numpy arrays.
        
        Args:
            value: The value to clip
            min_val: The minimum allowed value
            max_val: The maximum allowed value
            
        Returns:
            Clipped value as numpy array
        """
        try:
            # Ensure all inputs are numpy arrays
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            if not isinstance(min_val, np.ndarray):
                min_val = np.array(min_val)
            if not isinstance(max_val, np.ndarray):
                max_val = np.array(max_val)
                
            # Extract scalars if needed
            if isinstance(min_val, np.ndarray) and min_val.size == 1:
                min_val = min_val.item()
            if isinstance(max_val, np.ndarray) and max_val.size == 1:
                max_val = max_val.item()
                
            # Perform clip operation
            return np.clip(value, min_val, max_val)
        except Exception as e:
            print(f"Error in safe_clip: {e}")
            # Return original value if clip fails
            return value 

    def _safe_array(self, value):
        """
        Safely convert a value to a numpy array.
        
        Args:
            value: Value to convert (list, tuple, scalar, or numpy array)
            
        Returns:
            Value as a numpy array
        """
        try:
            if isinstance(value, np.ndarray):
                return value
            return np.array(value)
        except Exception as e:
            print(f"Error in safe_array conversion: {e}")
            # Fallback to empty array
            return np.array([0, 0])
    
    def _safe_compare(self, array1, array2, comparison_op='all_gte'):
        """
        Safely compare two arrays with specified comparison operation.
        
        Args:
            array1: First array
            array2: Second array
            comparison_op: Type of comparison ('all_gte', 'all_lte', 'all_gt', 'all_lt')
            
        Returns:
            Boolean result of comparison
        """
        try:
            # Ensure both are numpy arrays
            array1 = self._safe_array(array1)
            array2 = self._safe_array(array2)
            
            # Perform the comparison
            if comparison_op == 'all_gte':
                return np.all(array1 >= array2)
            elif comparison_op == 'all_lte':
                return np.all(array1 <= array2)
            elif comparison_op == 'all_gt':
                return np.all(array1 > array2)
            elif comparison_op == 'all_lt':
                return np.all(array1 < array2)
            else:
                return False
        except Exception as e:
            print(f"Error in safe_compare: {e}")
            return False 

    def _safe_normalize(self, vector):
        """
        Safely normalize a vector to unit length.
        
        Args:
            vector: Vector to normalize
            
        Returns:
            Normalized vector, or original if norm is zero
        """
        try:
            vector = self._safe_array(vector)
            norm = np.linalg.norm(vector)
            if norm > 0:
                return vector / norm
            return vector
        except Exception as e:
            print(f"Error in safe_normalize: {e}")
            return vector 