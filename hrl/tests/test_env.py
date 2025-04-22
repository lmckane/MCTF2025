import numpy as np
import matplotlib.pyplot as plt
from hrl.environment.game_env import GameEnvironment

def test_environment():
    """Test the game environment with random actions."""
    # Create environment
    config = {
        'map_size': [100, 100],
        'num_agents': 3,
        'max_steps': 1000,
        'tag_radius': 5,
        'capture_radius': 10,
        'base_radius': 20,
        'difficulty': 0.5
    }
    env = GameEnvironment(config)
    
    # Run test episodes
    num_episodes = 5
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        state = env.reset()
        done = False
        step = 0
        
        while not done:
            # Generate random actions for each agent
            actions = []
            for _ in range(len(env.agents)):
                # Random velocity in [-1, 1] range
                velocity = np.random.uniform(-1, 1, 2)
                actions.append(velocity)
                
            # Step environment
            next_state, reward, done, info = env.step(actions)
            
            # Render environment
            env.render()
            
            # Print step information
            print(f"Step {step + 1}:")
            print(f"  Reward: {reward}")
            print(f"  Flag captured: {info['flag_captured']}")
            print(f"  Tagged agents: {info['tagged']}")
            print(f"  Game state: {info['game_state']}")
            
            step += 1
            
        print(f"Episode finished after {step} steps")
        print(f"Final game state: {info['game_state']}")
        
    env.close()

def test_territory_checking():
    """Test the territory checking functionality."""
    config = {
        'map_size': [100, 100],
        'num_agents': 1,
        'max_steps': 1,
        'tag_radius': 5,
        'capture_radius': 10,
        'base_radius': 20,
        'difficulty': 0.5
    }
    env = GameEnvironment(config)
    
    # Test points in team 0 territory
    points_team0 = [
        np.array([10, 50]),  # Near base
        np.array([25, 25]),  # Corner
        np.array([25, 75]),  # Corner
        np.array([49, 50])   # Near center
    ]
    
    # Test points in team 1 territory
    points_team1 = [
        np.array([90, 50]),  # Near base
        np.array([75, 25]),  # Corner
        np.array([75, 75]),  # Corner
        np.array([51, 50])   # Near center
    ]
    
    print("\nTesting territory checking:")
    print("Team 0 territory:")
    for point in points_team0:
        in_team0 = env._is_in_territory(point, 0)
        in_team1 = env._is_in_territory(point, 1)
        print(f"  Point {point}: Team 0: {in_team0}, Team 1: {in_team1}")
        
    print("\nTeam 1 territory:")
    for point in points_team1:
        in_team0 = env._is_in_territory(point, 0)
        in_team1 = env._is_in_territory(point, 1)
        print(f"  Point {point}: Team 0: {in_team0}, Team 1: {in_team1}")
        
    env.close()

def visualize_territories():
    """Visualize the territories and test points."""
    config = {
        'map_size': [100, 100],
        'num_agents': 1,
        'max_steps': 1,
        'tag_radius': 5,
        'capture_radius': 10,
        'base_radius': 20,
        'difficulty': 0.5
    }
    env = GameEnvironment(config)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    
    # Draw territories
    for team, territory in env.territories.items():
        color = 'lightblue' if team == 0 else 'lightpink'
        ax.fill(*zip(*territory), color=color, alpha=0.3)
        
    # Draw team bases
    for team, base in env.team_bases.items():
        color = 'blue' if team == 0 else 'red'
        circle = plt.Circle(base, env.base_radius, color=color, alpha=0.5)
        ax.add_patch(circle)
        
    # Test points
    test_points = [
        (np.array([10, 50]), 'Team 0 Base'),
        (np.array([25, 25]), 'Team 0 Corner'),
        (np.array([25, 75]), 'Team 0 Corner'),
        (np.array([49, 50]), 'Team 0 Center'),
        (np.array([90, 50]), 'Team 1 Base'),
        (np.array([75, 25]), 'Team 1 Corner'),
        (np.array([75, 75]), 'Team 1 Corner'),
        (np.array([51, 50]), 'Team 1 Center')
    ]
    
    # Plot test points
    for point, label in test_points:
        in_team0 = env._is_in_territory(point, 0)
        in_team1 = env._is_in_territory(point, 1)
        color = 'green' if (in_team0 or in_team1) else 'red'
        ax.plot(point[0], point[1], 'o', color=color)
        ax.text(point[0], point[1] + 2, label, ha='center')
        
    plt.title('Territory Visualization')
    plt.show()
    env.close()

if __name__ == '__main__':
    print("Running environment tests...")
    test_environment()
    
    print("\nRunning territory checking tests...")
    test_territory_checking()
    
    print("\nVisualizing territories...")
    visualize_territories() 