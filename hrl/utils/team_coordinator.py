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
        Make initial role assignments based on the starting state.
        
        Args:
            state: Current game state
        """
        # Get agents on our team
        team_agents = [agent for agent in state['agents'] 
                       if agent['team'] == self.team_id]
        
        # Baseline role distribution (1 defender, rest attackers for small teams)
        if len(team_agents) <= 3:
            # With 3 or fewer agents, assign 1 defender and the rest attackers
            roles_to_assign = [AgentRole.DEFENDER] + [AgentRole.ATTACKER] * (len(team_agents) - 1)
        else:
            # With more agents, assign 1 defender, 1 interceptor, and the rest attackers
            roles_to_assign = [AgentRole.DEFENDER, AgentRole.INTERCEPTOR] + [AgentRole.ATTACKER] * (len(team_agents) - 2)
            
        # Shuffle roles to prevent predictable assignments
        random.shuffle(roles_to_assign)
        
        # Assign roles to agents
        self.agent_roles = {}
        for i, agent in enumerate(team_agents):
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
        
        # Adjusted role distribution based on game state
        if our_flag_captured:
            # Prioritize interceptors to recover our flag
            roles_to_assign = [AgentRole.INTERCEPTOR] * 2 + [AgentRole.ATTACKER] * (len(team_agents) - 2)
        elif we_have_enemy_flag:
            # Protect the flag carrier with defenders
            roles_to_assign = [AgentRole.DEFENDER] * 2 + [AgentRole.ATTACKER] * (len(team_agents) - 2)
        elif self.own_flag_threat > 0.7:  # High threat to our flag
            # Increase defense when our flag is threatened
            roles_to_assign = [AgentRole.DEFENDER] * 2 + [AgentRole.ATTACKER] * (len(team_agents) - 2)
        else:
            # Standard balanced distribution
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
        
        for agent in team_agents:
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
        for agent in team_agents:
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
                (agent['id'] if isinstance(agent, dict) else agent.id) in assigned_agents 
                for agent in team_agents
            ):
                break
                
            # Find best agent for this role
            best_agent_id = None
            best_score = -1
            
            for agent in team_agents:
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
        for agent in enemy_agents:
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
        for flag in state['flags']:
            flag_team = flag['team'] if isinstance(flag, dict) else flag.team
            flag_captured = flag['is_captured'] if isinstance(flag, dict) else flag.is_captured
            flag_carrier_id = flag['carrier_id'] if isinstance(flag, dict) else flag.carrier_id
            
            if flag_team == self.team_id and flag_captured:
                # Find the carrier in agents
                for enemy_agent in state['agents']:
                    enemy_id = enemy_agent['id'] if isinstance(enemy_agent, dict) else enemy_agent.id
                    if enemy_id == flag_carrier_id:
                        enemy_carrier = enemy_agent
                        break
        
        if enemy_carrier:
            # High priority to intercept flag carrier
            carrier_pos = np.array(enemy_carrier['position'] if isinstance(enemy_carrier, dict) else enemy_carrier.position)
            return 2.0 / (1.0 + np.linalg.norm(position - carrier_pos))
        
        # Otherwise, look for closest enemy in our territory
        enemy_in_territory = []
        for enemy in state['agents']:
            enemy_team = enemy['team'] if isinstance(enemy, dict) else enemy.team
            if enemy_team != self.team_id:
                enemy_pos = np.array(enemy['position'] if isinstance(enemy, dict) else enemy.position)
                # Check if enemy is in our territory
                if self._is_in_our_territory(enemy_pos, state):
                    distance = np.linalg.norm(position - enemy_pos)
                    enemy_in_territory.append((distance, enemy_pos))
        
        if enemy_in_territory:
            # Sort by distance
            enemy_in_territory.sort(key=lambda x: x[0])
            closest_enemy_distance, _ = enemy_in_territory[0]
            return 1.0 / (1.0 + closest_enemy_distance)
        
        # If no specific targets, lower suitability
        return 0.2
    
    def _is_in_our_territory(self, position: np.ndarray, state: Dict[str, Any]) -> bool:
        """
        Check if a position is in our territory.
        
        Args:
            position: Position to check
            state: Current game state
            
        Returns:
            True if position is in our territory
        """
        territories = state['territories']
        our_territory = territories[self.team_id]
        
        # Simple polygon check using ray casting algorithm
        inside = False
        n = len(our_territory)
        p1 = our_territory[0]
        
        for i in range(1, n + 1):
            p2 = our_territory[i % n]
            if position[1] > min(p1[1], p2[1]):
                if position[1] <= max(p1[1], p2[1]):
                    if position[0] <= max(p1[0], p2[0]):
                        if p1[1] != p2[1]:
                            xinters = (position[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                        if p1[0] == p2[0] or position[0] <= xinters:
                            inside = not inside
            p1 = p2
        
        return inside
    
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
        """Get position of our flag."""
        for flag in state['flags']:
            flag_team = flag['team'] if isinstance(flag, dict) else flag.team
            
            if flag_team == self.team_id:
                # If flag is captured, use base position instead
                flag_captured = flag['is_captured'] if isinstance(flag, dict) else flag.is_captured
                if flag_captured:
                    return np.array(state['team_bases'][self.team_id])
                else:
                    return np.array(flag['position'] if isinstance(flag, dict) else flag.position)
        
        # Fallback to base position if flag not found
        return np.array(state['team_bases'][self.team_id])
    
    def _get_enemy_flag_position(self, state: Dict[str, Any]) -> np.ndarray:
        """Get position of the enemy flag."""
        enemy_team = 1 - self.team_id  # Assuming two teams: 0 and 1
        
        for flag in state['flags']:
            flag_team = flag['team'] if isinstance(flag, dict) else flag.team
            
            if flag_team == enemy_team:
                # If flag is captured, find carrier
                flag_captured = flag['is_captured'] if isinstance(flag, dict) else flag.is_captured
                if flag_captured:
                    carrier_id = flag['carrier_id'] if isinstance(flag, dict) else flag.carrier_id
                    for agent in state['agents']:
                        agent_id = agent['id'] if isinstance(agent, dict) else agent.id
                        if agent_id == carrier_id:
                            return np.array(agent['position'] if isinstance(agent, dict) else agent.position)
                            
                    # If carrier not found, use enemy base as fallback
                    return np.array(state['team_bases'][enemy_team])
                else:
                    return np.array(flag['position'] if isinstance(flag, dict) else flag.position)
        
        # Fallback to enemy base position
        return np.array(state['team_bases'][enemy_team])
    
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
        for agent in state['agents']:
            agent_team = agent['team'] if isinstance(agent, dict) else agent.team
            if agent_team == self.team_id:
                team_agents.append(agent)
                
        # Get positions of all team members
        team_positions = {}
        team_roles = {}
        for agent in team_agents:
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
        for a in state['agents']:
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
                        perp_vector = perp_vector / np.linalg.norm(perp_vector)
                        
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
                        
                        # If waypoint is out of bounds, adjust
                        waypoint = np.clip(waypoint, [0, 0], map_size)
                        
                        # Return intermediate waypoint for safer path
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
                        patrol_dir = patrol_dir / np.linalg.norm(patrol_dir)
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
                    perp_vector = perp_vector / np.linalg.norm(perp_vector)
                    
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
                    
                    # Make sure point is within map bounds
                    flank_point = np.clip(flank_point, [0, 0], map_size)
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
                    carrier_direction = carrier_to_base / np.linalg.norm(carrier_to_base)
                    
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
                patrol_point = np.clip(patrol_point, [0, 0], map_size)
                return patrol_point
            else:
                # No clear threats - patrol around flag with dynamic pattern
                angle = (self.step_counter / 10.0) % (2 * np.pi)
                patrol_radius = 25
                
                # Create elliptical patrol that puts more emphasis on direction of enemy base
                direction_to_enemy = enemy_base - our_flag_pos
                if np.linalg.norm(direction_to_enemy) > 0:
                    direction_to_enemy = direction_to_enemy / np.linalg.norm(direction_to_enemy)
                    perpendicular = np.array([-direction_to_enemy[1], direction_to_enemy[0]])
                    
                    # Elongate patrol in the direction of potential threats
                    x_component = direction_to_enemy * patrol_radius * 1.5 * np.cos(angle)
                    y_component = perpendicular * patrol_radius * np.sin(angle)
                    
                    patrol_pos = our_flag_pos + x_component + y_component
                else:
                    # Standard circular patrol if no clear direction
                    patrol_pos = our_flag_pos + np.array([patrol_radius * np.cos(angle), patrol_radius * np.sin(angle)])
                
                # Make sure patrol point is within map bounds
                patrol_pos = np.clip(patrol_pos, [0, 0], map_size)
                return patrol_pos
        
        # Default: return to base
        return our_base 