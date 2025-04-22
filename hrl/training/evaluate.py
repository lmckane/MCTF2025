import numpy as np
import torch
import matplotlib.pyplot as plt
from hrl.training.trainer import Trainer
from hrl.environment.game_env import GameEnvironment

def evaluate_model(model_path: str, num_episodes: int = 100):
    """Evaluate a trained model."""
    # Load trainer with saved configuration
    trainer = Trainer.load_checkpoint(model_path)
    
    # Initialize metrics
    metrics = {
        'win_rate': [],
        'avg_score': [],
        'flag_captures': [],
        'tags': [],
        'deaths': [],
        'episode_lengths': []
    }
    
    # Run evaluation episodes
    print(f"Running {num_episodes} evaluation episodes...")
    for episode in range(num_episodes):
        # Create environment
        env = trainer._create_environment()
        state = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Get action from policy
            option = trainer.option_selector.select_option(state)
            action = trainer.policy.get_action(state, option)
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            
        # Record episode metrics
        metrics['win_rate'].append(1 if info['game_state'] == GameState.WON else 0)
        metrics['avg_score'].append(episode_reward)
        metrics['flag_captures'].append(info['flag_captured'])
        metrics['tags'].append(info['tagged'])
        metrics['deaths'].append(info['died'])
        metrics['episode_lengths'].append(episode_length)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")
            
    # Calculate and print statistics
    print("\nEvaluation Results:")
    print(f"Win Rate: {np.mean(metrics['win_rate']):.2f}")
    print(f"Average Score: {np.mean(metrics['avg_score']):.2f}")
    print(f"Average Flag Captures: {np.mean(metrics['flag_captures']):.2f}")
    print(f"Average Tags: {np.mean(metrics['tags']):.2f}")
    print(f"Average Deaths: {np.mean(metrics['deaths']):.2f}")
    print(f"Average Episode Length: {np.mean(metrics['episode_lengths']):.2f}")
    
    # Plot metrics
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(metrics['win_rate'])
    plt.title('Win Rate')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    
    plt.subplot(2, 3, 2)
    plt.plot(metrics['avg_score'])
    plt.title('Average Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(2, 3, 3)
    plt.plot(metrics['flag_captures'])
    plt.title('Flag Captures')
    plt.xlabel('Episode')
    plt.ylabel('Captures')
    
    plt.subplot(2, 3, 4)
    plt.plot(metrics['tags'])
    plt.title('Tags')
    plt.xlabel('Episode')
    plt.ylabel('Tags')
    
    plt.subplot(2, 3, 5)
    plt.plot(metrics['deaths'])
    plt.title('Deaths')
    plt.xlabel('Episode')
    plt.ylabel('Deaths')
    
    plt.subplot(2, 3, 6)
    plt.plot(metrics['episode_lengths'])
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.tight_layout()
    plt.savefig('evaluation_metrics.png')
    plt.close()
    
    return metrics

if __name__ == '__main__':
    # Evaluate the model
    model_path = 'final_model.pth'
    metrics = evaluate_model(model_path) 