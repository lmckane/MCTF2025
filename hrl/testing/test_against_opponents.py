import os
import argparse
import numpy as np
import torch
import time
import json
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from hrl.environment.game_env import GameEnvironment
from hrl.policies.hierarchical_policy import HierarchicalPolicy
from hrl.utils.option_selector import OptionSelector
from hrl.utils.state_processor import StateProcessor
from hrl.utils.metrics import Metrics
from hrl.utils.team_coordinator import TeamCoordinator, AgentRole
from hrl.utils.opponent_modeler import OpponentModeler

class OpponentTester:
    """Tests trained models against different opponent strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = config['model_path']
        self.render = config['render']
        self.episodes = config['episodes']
        self.opponent_strategy = config['opponent_strategy']
        self.difficulty = config['difficulty']
        self.debug_level = config['debug_level']
        
        print(f"Testing against '{self.opponent_strategy}' strategy with difficulty {self.difficulty}")
        
        # Initialize environment
        self.env = GameEnvironment({
            'map_size': [100, 100],
            'num_agents': 3,
            'max_steps': 500,
            'tag_radius': 10,
            'capture_radius': 15,
            'base_radius': 20,
            'difficulty': self.difficulty,
            'max_velocity': 5.0,
            'win_score': 3,
            'debug_level': self.debug_level
        })
        
        # Initialize state processor and option selector
        self.state_processor = StateProcessor({
            'options': ['capture_flag', 'return_to_base', 'tag_enemy', 'defend_base', 'intercept'],
            'debug_level': self.debug_level
        })
        
        self.option_selector = OptionSelector({
            'options': ['capture_flag', 'return_to_base', 'tag_enemy', 'defend_base', 'intercept'],
            'debug_level': self.debug_level
        })
        
        # Initialize coordination components
        self.team_coordinator = TeamCoordinator({
            'role_update_frequency': 5,
            'debug_level': self.debug_level
        })
        
        self.opponent_modeler = OpponentModeler({
            'history_length': 20,
            'debug_level': self.debug_level
        })
        
        # Initialize policy
        self.policy = HierarchicalPolicy(
            state_size=self.state_processor.state_size,
            action_size=2,  # 2D movement
            config={
                'learning_rate': 1e-4,
                'gamma': 0.99,
                'hidden_size': 256,
                'batch_size': 64,
                'buffer_size': 10000,
                'update_freq': 10,
                'options': ['capture_flag', 'return_to_base', 'tag_enemy', 'defend_base', 'intercept'],
                'debug_level': self.debug_level
            },
            option_selector=self.option_selector,
            state_processor=self.state_processor
        )
        
        # Load model
        self._load_model()
        
        # Initialize metrics
        self.metrics = Metrics()
        
        # Results storage
        self.results = {
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'scores': [],
            'rewards': [],
            'durations': []
        }
        
    def _load_model(self):
        """Load the trained model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        print(f"Loading model from {self.model_path}")
        checkpoint = torch.load(self.model_path)
        
        self.policy.load_state_dict(checkpoint['policy'])
        
        # Try loading option selector and state processor if available
        if 'option_selector' in checkpoint:
            self.option_selector.load_state_dict(checkpoint['option_selector'])
            
        if 'state_processor' in checkpoint and checkpoint['state_processor'] is not None:
            self.state_processor.load_state_dict(checkpoint['state_processor'])
            
        print("Model loaded successfully!")
            
    def run_tests(self):
        """Run tests against the specified opponent strategy."""
        print(f"Running {self.episodes} test episodes...")
        
        total_reward = 0
        total_score = 0
        total_duration = 0
        
        for episode in range(self.episodes):
            # Reset environment and coordination components
            state = self.env.reset()
            self.team_coordinator.reset()
            self.team_coordinator.assign_initial_roles(state)
            self.opponent_modeler.reset()
            
            # Run episode
            episode_reward, episode_info = self._run_episode(state)
            
            # Update metrics
            total_reward += episode_reward
            total_score += episode_info['team_scores'][0]
            total_duration += episode_info['steps']
            
            # Update results
            self.results['rewards'].append(episode_reward)
            self.results['scores'].append(episode_info['team_scores'][0])
            self.results['durations'].append(episode_info['steps'])
            
            if episode_info['result'] == 'win':
                self.results['wins'] += 1
            elif episode_info['result'] == 'loss':
                self.results['losses'] += 1
            else:
                self.results['draws'] += 1
                
            # Print episode results
            print(f"Episode {episode+1}/{self.episodes} - "
                 f"Result: {episode_info['result']}, "
                 f"Score: {episode_info['team_scores'][0]}-{episode_info['team_scores'][1]}, "
                 f"Reward: {episode_reward:.2f}, "
                 f"Duration: {episode_info['steps']} steps")
                
        # Print overall results
        print("\nTest Results:")
        print(f"Win Rate: {self.results['wins'] / self.episodes * 100:.1f}%")
        print(f"Average Score: {total_score / self.episodes:.2f}")
        print(f"Average Reward: {total_reward / self.episodes:.2f}")
        print(f"Average Duration: {total_duration / self.episodes:.1f} steps")
        
        # Save results
        self._save_results()
        
        # Visualize results if enabled
        if self.config.get('visualize', False):
            self._visualize_results()
            
    def _run_episode(self, state):
        """Run a single test episode."""
        done = False
        total_reward = 0
        step = 0
        max_steps = self.env.max_steps
        
        # Apply opponent strategy
        self._set_opponent_strategy()
        
        while not done and step < max_steps:
            # Enhance state with coordination and opponent data
            enhanced_state = self._enhance_state(state, step)
            
            # Get action from policy
            processed_state = self.policy.state_processor.process_state(enhanced_state)
            option, action, _ = self.policy.get_action(processed_state, deterministic=True)
            
            # Take action in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Update total reward
            total_reward += reward
            
            # Update coordination components
            self.team_coordinator.update_roles(next_state)
            self.opponent_modeler.update(next_state)
            
            # Render if enabled
            if self.render:
                self.env.render()
                time.sleep(0.05)  # Slow down rendering for visibility
                
            # Move to next state
            state = next_state
            step += 1
            
        # Create episode info
        episode_info = {
            'team_scores': self.env.team_scores,
            'steps': step,
            'result': 'win' if self.env.game_state.value == 1 else 'loss' if self.env.game_state.value == 2 else 'draw'
        }
        
        return total_reward, episode_info
        
    def _enhance_state(self, state, step):
        """Enhance state with coordination and opponent data."""
        enhanced_state = state.copy()
        
        # Add coordination data
        agent_id = 0  # Our agent
        coordination_data = self.team_coordinator.get_coordination_data(agent_id, state)
        enhanced_state['agent_role'] = coordination_data['agent_role'].value
        enhanced_state['recommended_target'] = coordination_data['recommended_target']
        enhanced_state['our_flag_threat'] = coordination_data['own_flag_threat']
        enhanced_state['our_flag_captured'] = coordination_data['our_flag_captured']
        enhanced_state['enemy_flag_captured'] = coordination_data['enemy_flag_captured']
        
        # Add opponent modeling data
        agent_pos = np.array(state['agents'][0]['position'])
        enemy_flag = next((f for f in state['flags'] if f['team'] != 0 and not f['is_captured']), None)
        
        if enemy_flag:
            flag_pos = np.array(enemy_flag['position'])
            safe_path = self.opponent_modeler.suggest_path(agent_pos, flag_pos, state)
            enhanced_state['suggested_flag_path'] = safe_path
            
        # Add danger score and counter strategy
        enhanced_state['position_danger'] = self.opponent_modeler.get_danger_score(agent_pos, state)
        enhanced_state['counter_strategy'] = self.opponent_modeler.get_counter_strategy()
        
        return enhanced_state
        
    def _set_opponent_strategy(self):
        """Set the opponent strategy based on configuration."""
        if self.opponent_strategy == "random":
            # No modifications needed - default random movement
            pass
            
        elif self.opponent_strategy == "direct":
            # Direct strategy - opponents move directly to objectives
            for i, agent in enumerate(self.env.agents):
                if i > 0:  # Only modify opponent agents
                    agent.strategy = "direct"
                    
        elif self.opponent_strategy == "defensive":
            # Defensive strategy - focus on defending flag
            for i, agent in enumerate(self.env.agents):
                if i > 0:  # Only modify opponent agents
                    agent.strategy = "defensive"
                    
        elif self.opponent_strategy == "aggressive":
            # Aggressive strategy - focus on tagging player agents
            for i, agent in enumerate(self.env.agents):
                if i > 0:  # Only modify opponent agents
                    agent.strategy = "aggressive"
                    
        elif self.opponent_strategy == "coordinated":
            # Coordinated strategy - agents work together
            # Set up opponent team roles
            for i, agent in enumerate(self.env.agents):
                if i > 0:  # Only modify opponent agents
                    agent.strategy = "coordinated"
                    # Assign roles: 1 defender, 1 interceptor, 1 attacker
                    if i == 1:
                        agent.role = AgentRole.DEFENDER.value
                    elif i == 2:
                        agent.role = AgentRole.INTERCEPTOR.value
                    else:
                        agent.role = AgentRole.ATTACKER.value
                        
    def _get_opponent_action(self, agent_idx):
        """Custom opponent action based on strategy."""
        agent = self.env.agents[agent_idx]
        strategy = getattr(agent, 'strategy', "random")
        
        if strategy == "direct":
            # Move directly toward objectives
            if agent.has_flag:
                # Return to base with flag
                target = self.env.team_bases[agent.team]
            else:
                # Go for enemy flag
                enemy_flags = [flag for flag in self.env.flags if flag.team != agent.team and not flag.is_captured]
                if enemy_flags:
                    target = enemy_flags[0].position
                else:
                    # No flags to capture, go to center
                    target = self.env.map_size / 2
                    
            direction = target - agent.position
            if np.linalg.norm(direction) > 0:
                return direction / np.linalg.norm(direction)
            return np.zeros(2)
            
        elif strategy == "defensive":
            # Stay near own flag or base
            own_flag = next((flag for flag in self.env.flags if flag.team == agent.team), None)
            
            if own_flag and not own_flag.is_captured:
                target = own_flag.position
            else:
                # Flag captured, defend base
                target = self.env.team_bases[agent.team]
                
            # Stay within defensive range
            direction = target - agent.position
            distance = np.linalg.norm(direction)
            
            if distance > 20:  # Only move if too far from defensive position
                if np.linalg.norm(direction) > 0:
                    return direction / np.linalg.norm(direction)
            elif distance < 15:  # Patrol behavior when close
                angle = np.random.uniform(0, 2 * np.pi)
                return np.array([np.cos(angle), np.sin(angle)])
                
            return np.zeros(2)
            
        elif strategy == "aggressive":
            # Hunt player agents
            player_agents = [a for a in self.env.agents if a.team == 0 and not a.is_tagged]
            
            if player_agents:
                # Target closest player agent
                target_agent = min(player_agents, key=lambda a: np.linalg.norm(agent.position - a.position))
                direction = target_agent.position - agent.position
                
                if np.linalg.norm(direction) > 0:
                    return direction / np.linalg.norm(direction)
            else:
                # No players to hunt, go to enemy base
                direction = self.env.team_bases[0] - agent.position
                if np.linalg.norm(direction) > 0:
                    return direction / np.linalg.norm(direction)
                    
            return np.zeros(2)
            
        elif strategy == "coordinated":
            # Coordinated team behavior based on roles
            role = getattr(agent, 'role', AgentRole.ATTACKER.value)
            
            if role == AgentRole.DEFENDER.value:
                # Defend flag or intercept if flag captured
                own_flag = next((flag for flag in self.env.flags if flag.team == agent.team), None)
                
                if own_flag and own_flag.is_captured:
                    # Flag captured - intercept carrier
                    carrier = next((a for a in self.env.agents if a.team == 0 and a.has_flag), None)
                    if carrier:
                        direction = carrier.position - agent.position
                    else:
                        # No carrier found, defend base
                        direction = self.env.team_bases[agent.team] - agent.position
                else:
                    # Flag not captured - defend it
                    target = own_flag.position if own_flag else self.env.team_bases[agent.team]
                    direction = target - agent.position
                    distance = np.linalg.norm(direction)
                    
                    if distance < 15:  # Patrol behavior when close
                        angle = np.random.uniform(0, 2 * np.pi)
                        return np.array([np.cos(angle), np.sin(angle)])
                        
            elif role == AgentRole.INTERCEPTOR.value:
                # Intercept enemies in territory
                player_agents = [a for a in self.env.agents if a.team == 0 and not a.is_tagged]
                
                if player_agents:
                    # Target closest enemy with highest priority to flag carriers
                    flag_carriers = [a for a in player_agents if a.has_flag]
                    
                    if flag_carriers:
                        target_agent = flag_carriers[0]
                    else:
                        # Target agent closest to our flag
                        own_flag = next((flag for flag in self.env.flags if flag.team == agent.team), None)
                        if own_flag:
                            target_agent = min(player_agents, 
                                             key=lambda a: np.linalg.norm(a.position - own_flag.position))
                        else:
                            target_agent = player_agents[0]
                            
                    direction = target_agent.position - agent.position
                else:
                    # No enemies to intercept, patrol center
                    direction = (self.env.map_size / 2) - agent.position
                    
            else:  # ATTACKER role
                # Attack to capture flag
                if agent.has_flag:
                    # Return to base with flag
                    direction = self.env.team_bases[agent.team] - agent.position
                else:
                    # Go for enemy flag
                    enemy_flags = [flag for flag in self.env.flags if flag.team != agent.team and not flag.is_captured]
                    if enemy_flags:
                        target = enemy_flags[0].position
                        
                        # Check if path is blocked by defenders
                        player_agents = [a for a in self.env.agents if a.team == 0 and not a.is_tagged]
                        if player_agents:
                            # If defender is close to flag, try to tag them first
                            defenders = [a for a in player_agents 
                                       if np.linalg.norm(a.position - target) < 20]
                            
                            if defenders:
                                # Target defender instead of flag
                                target = defenders[0].position
                                
                        direction = target - agent.position
                    else:
                        # No flags to capture, attack enemy base
                        direction = self.env.team_bases[0] - agent.position
                        
            # Normalize direction
            if np.linalg.norm(direction) > 0:
                return direction / np.linalg.norm(direction)
                
            return np.zeros(2)
            
        else:  # Default random behavior
            # Use built-in opponent AI
            return None
            
    def _save_results(self):
        """Save test results to a file."""
        results_file = f"results_{self.opponent_strategy}_diff{self.difficulty}.json"
        
        # Add summary statistics
        summary = {
            'opponent_strategy': self.opponent_strategy,
            'difficulty': self.difficulty,
            'episodes': self.episodes,
            'win_rate': self.results['wins'] / self.episodes,
            'average_score': sum(self.results['scores']) / self.episodes,
            'average_reward': sum(self.results['rewards']) / self.episodes,
            'average_duration': sum(self.results['durations']) / self.episodes
        }
        
        # Combine with detailed results
        output = {
            'summary': summary,
            'detailed': self.results
        }
        
        # Save to file
        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)
            
        print(f"Results saved to {results_file}")
        
    def _visualize_results(self):
        """Visualize test results."""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Win/Loss/Draw distribution
        plt.subplot(2, 2, 1)
        labels = ['Wins', 'Losses', 'Draws']
        sizes = [self.results['wins'], self.results['losses'], self.results['draws']]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(f'Outcome Distribution vs {self.opponent_strategy.capitalize()}')
        
        # Plot 2: Scores per episode
        plt.subplot(2, 2, 2)
        plt.plot(self.results['scores'], marker='o')
        plt.axhline(y=sum(self.results['scores']) / len(self.results['scores']), 
                   color='r', linestyle='--', label='Average')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Team 0 Score per Episode')
        plt.legend()
        
        # Plot 3: Rewards per episode
        plt.subplot(2, 2, 3)
        plt.plot(self.results['rewards'], marker='o')
        plt.axhline(y=sum(self.results['rewards']) / len(self.results['rewards']), 
                   color='r', linestyle='--', label='Average')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward per Episode')
        plt.legend()
        
        # Plot 4: Episode durations
        plt.subplot(2, 2, 4)
        plt.plot(self.results['durations'], marker='o')
        plt.axhline(y=sum(self.results['durations']) / len(self.results['durations']), 
                   color='r', linestyle='--', label='Average')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Episode Duration')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"results_{self.opponent_strategy}_diff{self.difficulty}.png")
        plt.show()
        
def parse_args():
    parser = argparse.ArgumentParser(description='Test trained model against different opponent strategies')
    parser.add_argument('--model', type=str, default='final_model.pth', help='Path to model file')
    parser.add_argument('--opponent', type=str, default='random', 
                      choices=['random', 'direct', 'defensive', 'aggressive', 'coordinated'],
                      help='Opponent strategy to test against')
    parser.add_argument('--difficulty', type=float, default=0.9, help='Difficulty level (0.0-1.0)')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to test')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--debug_level', type=int, default=1, help='Debug level (0=none, 1=minimal, 2=verbose)')
    return parser.parse_args()
    
def main():
    args = parse_args()
    
    # Create configuration
    config = {
        'model_path': args.model,
        'opponent_strategy': args.opponent,
        'difficulty': args.difficulty,
        'episodes': args.episodes,
        'render': args.render,
        'visualize': args.visualize,
        'debug_level': args.debug_level
    }
    
    # Initialize tester
    tester = OpponentTester(config)
    
    # Run tests
    tester.run_tests()
    
if __name__ == '__main__':
    main() 