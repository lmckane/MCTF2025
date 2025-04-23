import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import random

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hrl.utils.team_coordinator import TeamCoordinator, AgentRole
from hrl.utils.state_processor import StateProcessor
from hrl.environment.game_env import GameEnvironment, GameState

def create_test_state(env):
    """Create a test state from the environment."""
    state = env._get_observation()
    
    # Add agent IDs if not present
    for i, agent in enumerate(state['agents']):
        if 'id' not in agent:
            agent['id'] = i
            
    return state

def create_complex_state(env):
    """Create a more complex state with various game scenarios."""
    # First create a base state
    state = create_test_state(env)
    
    # Modify it to create a more interesting scenario
    # 1. Move some agents to strategic positions
    for i, agent in enumerate(state['agents']):
        if agent['team'] == 0:  # Our team
            if i == 0:
                # First agent near enemy flag
                enemy_flag = next(flag for flag in state['flags'] if flag['team'] == 1)
                agent['position'] = enemy_flag['position'] - np.array([10, 0])
            elif i == 1:
                # Second agent in the middle
                agent['position'] = np.array(env.map_size) / 2
            else:
                # Others near our base
                our_base = state['team_bases'][0]
                agent['position'] = our_base + np.random.uniform(-15, 15, 2)
        else:  # Enemy team
            if i == 3:  # First enemy agent approaching our flag
                our_flag = next(flag for flag in state['flags'] if flag['team'] == 0)
                agent['position'] = our_flag['position'] + np.array([15, 5])
            elif i == 4:  # Second enemy agent has our flag
                our_flag = next(flag for flag in state['flags'] if flag['team'] == 0)
                our_flag['is_captured'] = True
                our_flag['carrier_id'] = agent['id']
                agent['has_flag'] = True
                agent['position'] = our_flag['position'] + np.array([20, 0])
            else:
                # Others randomly positioned
                agent['position'] = np.random.uniform([0, 0], env.map_size, 2)
    
    return state

def visualize_coordination(state, team_coordinator, role_colors=None):
    """Visualize the team coordination."""
    if role_colors is None:
        role_colors = {
            AgentRole.ATTACKER: 'blue',
            AgentRole.DEFENDER: 'green',
            AgentRole.INTERCEPTOR: 'purple'
        }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    map_size = state.get('map_size', [100, 100])
    ax.set_xlim(0, map_size[0])
    ax.set_ylim(0, map_size[1])
    
    # Draw territories
    for team, territory in state['territories'].items():
        color = 'lightblue' if team == 0 else 'lightpink'
        polygon = Polygon(territory, alpha=0.3, color=color)
        ax.add_patch(polygon)
    
    # Draw bases
    for team, base_pos in state['team_bases'].items():
        color = 'blue' if team == 0 else 'red'
        base_circle = Circle(base_pos, 20, alpha=0.3, color=color)
        ax.add_patch(base_circle)
    
    # Draw flags
    for flag in state['flags']:
        color = 'blue' if flag['team'] == 0 else 'red'
        if not flag['is_captured']:
            ax.plot(flag['position'][0], flag['position'][1], 'v', markersize=12, color=color)
    
    # Draw agents
    for i, agent in enumerate(state['agents']):
        agent_id = agent.get('id', i)
        team = agent['team']
        position = agent['position']
        
        # Decide marker and color
        if team == 0:  # Our team
            role = team_coordinator.get_agent_role(agent_id)
            color = role_colors.get(role, 'blue')
            marker = 'o'
        else:  # Enemy team
            color = 'red'
            marker = 's'  # square for enemies
        
        # Make size larger for flag carriers
        size = 12 if agent.get('has_flag', False) else 8
        
        # Special marker for tagged agents
        if agent.get('is_tagged', False):
            marker = 'x'
        
        # Plot agent
        ax.plot(position[0], position[1], marker, color=color, markersize=size)
        
        # Add agent ID
        ax.text(position[0] + 2, position[1] + 2, str(agent_id), color='black')
        
        # If our team agent, also draw recommended target
        if team == 0:
            role = team_coordinator.get_agent_role(agent_id)
            coordination_data = team_coordinator.get_coordination_data(agent_id, state)
            target = coordination_data['recommended_target']
            
            # Draw line to target
            ax.plot([position[0], target[0]], [position[1], target[1]], '--', color=color, alpha=0.5)
            
            # Add role label
            role_name = role.name if hasattr(role, 'name') else str(role)
            ax.text(position[0], position[1] - 5, role_name, color=color, ha='center')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Attacker'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Defender'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Interceptor'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Enemy'),
        Line2D([0], [0], marker='v', color='blue', markersize=10, label='Our Flag'),
        Line2D([0], [0], marker='v', color='red', markersize=10, label='Enemy Flag')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add title
    our_flag_status = "Captured" if any(f['team'] == 0 and f['is_captured'] for f in state['flags']) else "Safe"
    enemy_flag_status = "Captured" if any(f['team'] == 1 and f['is_captured'] for f in state['flags']) else "Safe"
    ax.set_title(f"Team Coordination Visualization\nOur Flag: {our_flag_status}, Enemy Flag: {enemy_flag_status}")
    
    # Show threat levels
    threat_text = f"Our Flag Threat: {team_coordinator.own_flag_threat:.2f}"
    ax.text(5, 5, threat_text, fontsize=10)
    
    return fig

def test_team_coordination():
    """Test the TeamCoordinator class functionality."""
    # Create environment
    env_config = {
        'map_size': [100, 100],
        'num_agents': 3,
        'max_steps': 500,
        'tag_radius': 5,
        'capture_radius': 10,
        'base_radius': 20,
        'debug_level': 0
    }
    env = GameEnvironment(env_config)
    
    # Create team coordinator
    coordinator_config = {
        'num_agents': env_config['num_agents'],
        'role_update_frequency': 10,
        'team_id': 0  # Our team
    }
    team_coordinator = TeamCoordinator(coordinator_config)
    
    # Create test state
    state = create_test_state(env)
    
    # Test initial role assignment
    team_coordinator.assign_initial_roles(state)
    print("Initial role assignment:")
    for agent in state['agents']:
        agent_id = agent.get('id', 0)
        if agent['team'] == 0:  # Our team
            role = team_coordinator.get_agent_role(agent_id)
            print(f"  Agent {agent_id}: {role.name if hasattr(role, 'name') else role}")
    
    # Test role update based on game state
    complex_state = create_complex_state(env)
    team_coordinator.update_roles(complex_state)
    print("\nUpdated roles based on complex state:")
    for agent in complex_state['agents']:
        agent_id = agent.get('id', 0)
        if agent['team'] == 0:  # Our team
            role = team_coordinator.get_agent_role(agent_id)
            print(f"  Agent {agent_id}: {role.name if hasattr(role, 'name') else role}")
    
    # Test coordination data
    for agent in complex_state['agents']:
        agent_id = agent.get('id', 0)
        if agent['team'] == 0:  # Our team
            coordination_data = team_coordinator.get_coordination_data(agent_id, complex_state)
            role = coordination_data['agent_role']
            target = coordination_data['recommended_target']
            print(f"\nCoordination data for Agent {agent_id} ({role.name if hasattr(role, 'name') else role}):")
            print(f"  Recommended target: {target}")
            print(f"  Our flag captured: {coordination_data['our_flag_captured']}")
            print(f"  Enemy flag captured: {coordination_data['enemy_flag_captured']}")
            print(f"  Our flag threat level: {coordination_data['own_flag_threat']:.2f}")
    
    # Visualize coordination
    fig = visualize_coordination(complex_state, team_coordinator)
    plt.savefig("team_coordination_test.png")
    print("\nVisualization saved to team_coordination_test.png")
    
    return team_coordinator, complex_state

def test_state_processor_integration():
    """Test integration with StateProcessor."""
    # Create environment
    env_config = {
        'map_size': [100, 100],
        'num_agents': 3,
        'max_steps': 500,
        'tag_radius': 5,
        'capture_radius': 10,
        'base_radius': 20,
        'debug_level': 0
    }
    env = GameEnvironment(env_config)
    
    # Create state processor with team coordination
    processor = StateProcessor(env_config)
    
    # Create test state
    state = create_complex_state(env)
    
    # Process state with team coordination
    processed_state = processor.process_state(state)
    
    # Print some processed state attributes
    print("\nProcessed State with Team Coordination:")
    print(f"  Agent roles: {processed_state.agent_roles}")
    print(f"  Recommended target: {processed_state.recommended_target}")
    print(f"  Our flag threat: {processed_state.our_flag_threat}")
    print(f"  Our flag captured: {processed_state.our_flag_captured}")
    print(f"  Enemy flag captured: {processed_state.enemy_flag_captured}")
    
    return processed_state

def test_dynamic_role_changes():
    """Test how roles change over time based on game state."""
    # Create environment
    env_config = {
        'map_size': [100, 100],
        'num_agents': 3,
        'max_steps': 500,
        'tag_radius': 5,
        'capture_radius': 10,
        'base_radius': 20,
        'debug_level': 0
    }
    env = GameEnvironment(env_config)
    
    # Create team coordinator
    coordinator_config = {
        'num_agents': env_config['num_agents'],
        'role_update_frequency': 1,  # Update every step for testing
        'team_id': 0  # Our team
    }
    team_coordinator = TeamCoordinator(coordinator_config)
    
    # Create initial state
    state = create_test_state(env)
    
    # Simulate a sequence of game states
    print("\nRole adaptations over time:")
    print("===========================")
    
    # Store agent roles for tracking
    agent_roles = {}
    
    # Record state history for each step
    state_history = []
    
    # Step 1: Normal state
    team_coordinator.update_roles(state)
    state_history.append(("Normal state", state.copy()))
    print("\nStep 1: Normal state")
    for agent in state['agents']:
        agent_id = agent.get('id', 0)
        if agent['team'] == 0:  # Our team
            role = team_coordinator.get_agent_role(agent_id)
            agent_roles[agent_id] = role
            print(f"  Agent {agent_id}: {role.name if hasattr(role, 'name') else role}")
    
    # Step 2: Enemy approaching our flag
    state2 = state.copy()
    our_flag = next(flag for flag in state2['flags'] if flag['team'] == 0)
    for agent in state2['agents']:
        if agent['team'] == 1:  # Enemy team
            # Move enemy agent close to our flag
            agent['position'] = our_flag['position'] + np.array([10, 0])
            break
    
    team_coordinator.update_roles(state2)
    state_history.append(("Enemy near our flag", state2.copy()))
    print("\nStep 2: Enemy approaching our flag")
    for agent in state2['agents']:
        agent_id = agent.get('id', 0)
        if agent['team'] == 0:  # Our team
            role = team_coordinator.get_agent_role(agent_id)
            print(f"  Agent {agent_id}: {role.name if hasattr(role, 'name') else role} (was {agent_roles[agent_id].name if agent_id in agent_roles and hasattr(agent_roles[agent_id], 'name') else agent_roles.get(agent_id, 'Unknown')})")
            agent_roles[agent_id] = role
    
    # Step 3: Enemy captured our flag
    state3 = state2.copy()
    our_flag = next(flag for flag in state3['flags'] if flag['team'] == 0)
    our_flag['is_captured'] = True
    for agent in state3['agents']:
        if agent['team'] == 1:  # Enemy team
            our_flag['carrier_id'] = agent.get('id', 0)
            agent['has_flag'] = True
            break
    
    team_coordinator.update_roles(state3)
    state_history.append(("Our flag captured", state3.copy()))
    print("\nStep 3: Enemy captured our flag")
    for agent in state3['agents']:
        agent_id = agent.get('id', 0)
        if agent['team'] == 0:  # Our team
            role = team_coordinator.get_agent_role(agent_id)
            print(f"  Agent {agent_id}: {role.name if hasattr(role, 'name') else role} (was {agent_roles[agent_id].name if agent_id in agent_roles and hasattr(agent_roles[agent_id], 'name') else agent_roles.get(agent_id, 'Unknown')})")
            agent_roles[agent_id] = role
    
    # Step 4: We captured enemy flag
    state4 = state3.copy()
    enemy_flag = next(flag for flag in state4['flags'] if flag['team'] == 1)
    enemy_flag['is_captured'] = True
    for agent in state4['agents']:
        if agent['team'] == 0:  # Our team
            enemy_flag['carrier_id'] = agent.get('id', 0)
            agent['has_flag'] = True
            break
    
    team_coordinator.update_roles(state4)
    state_history.append(("We captured enemy flag", state4.copy()))
    print("\nStep 4: We captured enemy flag")
    for agent in state4['agents']:
        agent_id = agent.get('id', 0)
        if agent['team'] == 0:  # Our team
            role = team_coordinator.get_agent_role(agent_id)
            print(f"  Agent {agent_id}: {role.name if hasattr(role, 'name') else role} (was {agent_roles[agent_id].name if agent_id in agent_roles and hasattr(agent_roles[agent_id], 'name') else agent_roles.get(agent_id, 'Unknown')})")
            agent_roles[agent_id] = role
    
    # Visualize different states
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, (title, state) in enumerate(state_history):
        # Set current axes
        plt.sca(axes[i])
        # Create visualization
        visualize_coordination(state, team_coordinator)
        axes[i].set_title(f"Step {i+1}: {title}")
    
    plt.tight_layout()
    plt.savefig("team_coordination_sequence.png")
    print("\nVisualization of role changes saved to team_coordination_sequence.png")
    
    return team_coordinator, state_history

if __name__ == "__main__":
    print("Testing Team Coordination implementation...")
    coordinator, state = test_team_coordination()
    processed_state = test_state_processor_integration()
    dynamic_coordinator, state_history = test_dynamic_role_changes()
    print("\nAll tests completed.") 