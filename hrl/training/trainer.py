import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import random
import time

from hrl.utils.metrics import MetricsTracker
from hrl.policies.hierarchical_policy import HierarchicalPolicy, Experience
from hrl.utils.option_selector import OptionSelector
from hrl.utils.state_processor import StateProcessor, ProcessedState
from hrl.environment.game_env import GameEnvironment, Agent, GameState

class Trainer:
    """Handles the training process with curriculum learning and adversarial training."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary containing:
                - env_config: Environment configuration
                - policy_config: Policy configuration
                - training_config: Training parameters
                - curriculum_config: Curriculum learning parameters
                - adversarial_config: Adversarial training parameters
        """
        self.config = config
        self.env_config = config['env_config']
        self.policy_config = config['policy_config']
        self.training_config = config['training_config']
        self.curriculum_config = config['curriculum_config']
        self.adversarial_config = config['adversarial_config']
        
        # Initialize components
        self.policy = HierarchicalPolicy(self.policy_config)
        self.option_selector = OptionSelector(self.policy_config)
        self.state_processor = StateProcessor(self.env_config)
        self.metrics = MetricsTracker(self.training_config.get('metrics_config', {}))
        
        # Training parameters
        self.num_episodes = self.training_config.get('num_episodes', 10000)
        self.max_steps = self.training_config.get('max_steps', 1000)
        self.batch_size = self.training_config.get('batch_size', 32)
        self.gamma = self.training_config.get('gamma', 0.99)
        self.lambda_ = self.training_config.get('lambda', 0.95)
        self.entropy_coef = self.training_config.get('entropy_coef', 0.01)
        self.learning_rate = self.training_config.get('learning_rate', 0.001)
        self.log_interval = self.training_config.get('log_interval', 100)
        self.checkpoint_interval = self.training_config.get('checkpoint_interval', 1000)
        self.render = self.training_config.get('render', False)
        self.debug_level = self.training_config.get('debug_level', 1)
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_wins = []
        self.episode_scores = []
        
        # Curriculum parameters
        self.curriculum_stages = self.curriculum_config.get('stages', [
            {'name': 'basic', 'difficulty': 0.2, 'duration': 0.2},
            {'name': 'intermediate', 'difficulty': 0.5, 'duration': 0.3},
            {'name': 'advanced', 'difficulty': 0.8, 'duration': 0.5}
        ])
        self.current_stage = 0
        self.stage_progress = 0.0
        
        # Adversarial parameters
        self.adversarial_ratio = self.adversarial_config.get('ratio', 0.3)
        self.adversarial_update_freq = self.adversarial_config.get('update_freq', 100)
        self.adversarial_steps = 0
        
        # Create log directory
        self.log_dir = os.path.join(
            self.training_config.get('log_dir', 'logs'),
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
    def train(self):
        """Run training loop."""
        print(f"\n{'='*80}")
        print(f"Starting training with {self.num_episodes} episodes")
        print(f"{'='*80}")
        
        total_episodes = 0
        start_time = time.time()
        
        for stage in range(3):  # Basic, intermediate, advanced
            print(f"\n{'-'*60}")
            print(f"Starting training stage {stage + 1}: {self.curriculum_stages[stage]['name']}")
            print(f"Difficulty: {self.curriculum_stages[stage]['difficulty']}")
            print(f"{'-'*60}")
            
            # Update curriculum stage
            self.current_stage = stage
            self.stage_progress = 0.0
            
            # Calculate episodes for this stage
            stage_duration = self.curriculum_stages[stage]['duration']
            episodes_in_stage = int(self.num_episodes * stage_duration)
            
            for episode in range(episodes_in_stage):
                total_episodes += 1
                
                # Create environment for this episode
                env = self._create_environment()
                env.debug_level = max(0, self.debug_level - 1)  # Reduce env verbosity
                
                # Run episode
                episode_start = time.time()
                experiences = self._run_episode(env)
                episode_time = time.time() - episode_start
                
                # Update policy
                self._update_policy(experiences)
                
                # Track episode performance
                total_reward = sum(exp.reward for exp in experiences)
                self.episode_rewards.append(total_reward)
                self.episode_lengths.append(len(experiences))
                
                # Track win/loss
                game_state = experiences[-1].state.get('game_state', GameState.PLAYING)
                win = 1.0 if game_state == GameState.WON else 0.0
                self.episode_wins.append(win)
                
                # Track team scores
                if 'agents' in experiences[-1].state:
                    self.episode_scores.append(env.team_scores[0])
                
                # Log metrics
                if episode % self.log_interval == 0 or episode == episodes_in_stage - 1:
                    self._log_detailed_metrics(episode, total_episodes, stage, episode_time)
                
                # Print progress
                if episode % max(1, episodes_in_stage // 100) == 0:
                    self._print_progress(episode, episodes_in_stage, total_episodes, stage)
                
                # Save checkpoint
                if episode % self.checkpoint_interval == 0:
                    self.save_checkpoint(f"stage_{stage}_episode_{episode}")
                    
                # Update adversarial opponent
                if episode % self.adversarial_update_freq == 0:
                    self._update_adversarial_opponent()
                    
                # Close environment
                env.close()
        
        # Training completed
        training_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"{'='*80}")
        
        # Save final model and metrics
        self.save_checkpoint(f"final_model")
        self.metrics.save_metrics(f"{self.log_dir}/final_metrics.json")
        
    def _print_progress(self, episode, episodes_in_stage, total_episodes, stage):
        """Print training progress in a concise format."""
        # Calculate recent statistics
        recent_rewards = self.episode_rewards[-100:]
        recent_wins = self.episode_wins[-100:]
        recent_scores = self.episode_scores[-100:] if self.episode_scores else [0]
        
        # Print progress bar and stats
        progress = episode / episodes_in_stage * 100
        stage_name = self.curriculum_stages[stage]['name']
        
        print(f"\r[Stage {stage+1}/{3}] {stage_name} "
              f"Progress: {episode}/{episodes_in_stage} ({progress:.1f}%) "
              f"Win Rate: {np.mean(recent_wins):.2f} "
              f"Avg Reward: {np.mean(recent_rewards):.2f} "
              f"Avg Score: {np.mean(recent_scores):.2f}", end="")
              
        # Print newline every 10 iterations for readability
        if episode % (episodes_in_stage // 10) == 0 and episode > 0:
            print()
        
        # Flush to ensure output is displayed
        import sys
        sys.stdout.flush()
        
    def _log_detailed_metrics(self, episode, total_episodes, stage, episode_time):
        """Log detailed metrics and print summary."""
        # Get recent metrics
        recent_rewards = self.episode_rewards[-100:]
        recent_wins = self.episode_wins[-100:]
        recent_lengths = self.episode_lengths[-100:]
        recent_scores = self.episode_scores[-100:] if self.episode_scores else [0]
        
        # Log to metrics tracker
        self.metrics.log_training_metric('episode', total_episodes)
        self.metrics.log_training_metric('stage', stage)
        self.metrics.log_game_metric('win_rate', np.mean(recent_wins))
        self.metrics.log_training_metric('avg_reward', np.mean(recent_rewards))
        self.metrics.log_training_metric('episode_length', np.mean(recent_lengths))
        self.metrics.log_game_metric('avg_score', np.mean(recent_scores))
        
        # Print detailed summary
        if self.debug_level >= 1:
            print(f"\n{'-'*80}")
            print(f"Episode {total_episodes} Summary (Stage {stage+1})")
            print(f"Time: {episode_time:.2f}s, Steps: {recent_lengths[-1]}")
            print(f"Recent Stats (last 100 episodes):")
            print(f"  Win Rate: {np.mean(recent_wins):.2f}")
            print(f"  Avg Reward: {np.mean(recent_rewards):.2f} (min={np.min(recent_rewards):.2f}, max={np.max(recent_rewards):.2f})")
            print(f"  Avg Score: {np.mean(recent_scores):.2f}")
            print(f"  Avg Episode Length: {np.mean(recent_lengths):.2f}")
            
            # Print option usage if available
            if hasattr(self, 'policy') and hasattr(self.policy, 'buffer'):
                recent_batch = self.policy.buffer.sample(min(100, len(self.policy.buffer)))
                if recent_batch:
                    option_counts = {}
                    for exp in recent_batch:
                        if not hasattr(exp, 'option'):
                            continue
                        opt = exp.option
                        option_counts[opt] = option_counts.get(opt, 0) + 1
                    
                    if option_counts:
                        print("  Option Usage:")
                        for opt, count in option_counts.items():
                            usage_percent = count / len(recent_batch) * 100
                            print(f"    {opt}: {usage_percent:.1f}%")
            
            print(f"{'-'*80}")
        
        # Save metrics
        if episode % (self.log_interval * 5) == 0:
            self.metrics.save_metrics(f"{self.log_dir}/metrics_episode_{total_episodes}.json")
            
    def _create_environment(self):
        """Create environment with current curriculum settings."""
        # Get current stage difficulty
        current_stage = self.curriculum_stages[self.current_stage]
        difficulty = current_stage['difficulty']
        
        # Adjust environment parameters based on difficulty
        env_config = self.env_config.copy()
        env_config['difficulty'] = difficulty
        env_config['debug_level'] = self.debug_level  # Pass debug level to environment
        
        # Adjust parameters based on difficulty
        env_config['num_agents'] = int(3 + difficulty * 2)  # More agents at higher difficulty
        env_config['tag_radius'] = 5 - difficulty * 2  # Smaller tag radius at higher difficulty
        env_config['capture_radius'] = 10 - difficulty * 3  # Smaller capture radius at higher difficulty
        
        # Create environment
        return GameEnvironment(env_config)
        
    def _run_episode(self, env: GameEnvironment) -> Dict[str, Any]:
        """Run a single training episode."""
        state = env.reset()
        processed_state = self.state_processor.process_state(state)
        experiences = []
        episode_info = {
            'rewards': [],
            'steps': 0,
            'done': False
        }
        
        while not episode_info['done']:
            # Select option
            option = self.option_selector.select_option(processed_state)
            
            # Get action from policy
            action = self.policy.get_action(processed_state, option)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            processed_next_state = self.state_processor.process_state(next_state)
            
            # Store experience
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                processed_state=processed_state,
                processed_next_state=processed_next_state,
                option=option  
            )
            experiences.append(experience)
            
            # Update state
            state = next_state
            processed_state = processed_next_state
            
            # Update episode info
            episode_info['rewards'].append(reward)
            episode_info['steps'] += 1
            episode_info['done'] = done
            
            # Render if enabled
            if self.render:
                env.render()
                
        return experiences
        
    def _update_policy(self, experiences: List[Experience]):
        """Update the policy using collected experiences."""
        # Store experiences in buffer
        for exp in experiences:
            self.policy.buffer.store(
                state=exp.state,
                processed_state=exp.processed_state,
                action=exp.action,
                reward=exp.reward,
                next_state=exp.next_state,
                processed_next_state=exp.processed_next_state,
                done=exp.done,
                option=exp.option
            )
            
        # Sample batch and update policy
        if len(self.policy.buffer) >= self.batch_size:
            batch = self.policy.buffer.sample(self.batch_size)
            
            # Compute advantages
            advantages = []
            last_gae = 0
            
            # Convert ProcessedState objects to tensors
            def state_to_tensor(state: ProcessedState) -> torch.Tensor:
                return torch.tensor([
                    state.agent_positions[0][0], state.agent_positions[0][1],
                    state.agent_velocities[0][0], state.agent_velocities[0][1],
                    state.agent_flags[0],
                    state.agent_tags[0],
                    state.agent_health[0],
                    state.agent_teams[0]
                ], dtype=torch.float32)
            
            # Convert batch to tensors
            states = torch.stack([state_to_tensor(exp.processed_state) for exp in batch])
            rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32)
            next_states = torch.stack([state_to_tensor(exp.processed_next_state) for exp in batch])
            dones = torch.tensor([exp.done for exp in batch], dtype=torch.float32)
            
            # Get values
            with torch.no_grad():
                _, values = self.policy.option_networks[batch[0].option](states)
                _, next_values = self.policy.option_networks[batch[0].option](next_states)
                values = values.squeeze()
                next_values = next_values.squeeze()
                
            # Compute GAE
            for t in reversed(range(len(batch))):
                if t == len(batch) - 1:
                    next_value = 0 if dones[t] else next_values[t]
                else:
                    next_value = values[t + 1]
                    
                delta = rewards[t] + self.gamma * next_value - values[t]
                last_gae = delta + self.gamma * self.lambda_ * (1 - dones[t]) * last_gae
                advantages.insert(0, last_gae)
                
            advantages = torch.tensor(advantages, dtype=torch.float32)
            
            # Update policy
            self.policy.update(batch, advantages)
            
        # Update option selector
        self.option_selector.update_weights(experiences)
        
    def _compute_advantages(self, batch: List[Experience]) -> torch.Tensor:
        """Compute advantages using GAE."""
        advantages = []
        last_gae = 0
        
        # Convert batch to tensors
        states = torch.stack([self.state_processor.process_state(exp.state) for exp in batch])
        rewards = torch.tensor([exp.reward for exp in batch], device=self.device)
        next_states = torch.stack([self.state_processor.process_state(exp.next_state) for exp in batch])
        dones = torch.tensor([exp.done for exp in batch], device=self.device)
        
        # Get values
        with torch.no_grad():
            _, values = self.policy.option_networks[batch[0].option](states)
            _, next_values = self.policy.option_networks[batch[0].option](next_states)
            values = values.squeeze()
            next_values = next_values.squeeze()
            
        # Compute GAE
        for t in reversed(range(len(batch))):
            if t == len(batch) - 1:
                next_value = 0 if dones[t] else next_values[t]
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.lambda_ * (1 - dones[t]) * last_gae
            advantages.insert(0, last_gae)
            
        return torch.tensor(advantages, device=self.device)
        
    def _update_curriculum_stage(self, episode: int):
        """Update the current curriculum stage based on progress and performance."""
        total_episodes = self.num_episodes
        current_progress = episode / total_episodes
        
        # Get recent performance metrics
        recent_wins = self.metrics.metrics['win_rate'][-100:] if self.metrics.metrics['win_rate'] else []
        win_rate = np.mean(recent_wins) if recent_wins else 0.0
        
        # Find current stage
        stage_progress = 0.0
        for i, stage in enumerate(self.curriculum_stages):
            stage_progress += stage['duration']
            if current_progress <= stage_progress:
                # Check if ready to advance
                if i == self.current_stage and win_rate > 0.7:  # 70% win rate threshold
                    self.current_stage = min(i + 1, len(self.curriculum_stages) - 1)
                    print(f"Advancing to stage {self.current_stage} with win rate {win_rate:.2f}")
                elif i == self.current_stage and win_rate < 0.3:  # 30% win rate threshold
                    self.current_stage = max(i - 1, 0)
                    print(f"Regressing to stage {self.current_stage} with win rate {win_rate:.2f}")
                    
                self.stage_progress = (current_progress - (stage_progress - stage['duration'])) / stage['duration']
                break
                
    def _update_adversarial_opponent(self):
        """Update the adversarial opponent policy."""
        # Check if we should update
        if random.random() > self.adversarial_ratio:
            return
            
        # Sample adversarial experiences
        if len(self.policy.buffer) >= self.batch_size:
            batch = self.policy.buffer.sample_adversarial(self.batch_size)
            self.policy.update_adversarial(batch)
            
    def _get_adversarial_action(self, state: Dict[str, Any], agent: Agent) -> np.ndarray:
        """Get action for adversarial agent."""
        # Simple adversarial strategy
        if agent.has_flag:
            # Return to base
            base_pos = self.team_bases[agent.team]
            direction = base_pos - agent.position
        else:
            # Chase nearest opponent
            nearest_opponent = None
            min_dist = float('inf')
            for other in self.agents:
                if other.team != agent.team and not other.is_tagged:
                    dist = np.linalg.norm(agent.position - other.position)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_opponent = other
                        
            if nearest_opponent:
                direction = nearest_opponent.position - agent.position
            else:
                # Move towards opponent's flag
                opponent_flag = next(flag for flag in self.flags if flag.team != agent.team)
                direction = opponent_flag.position - agent.position
                
        # Normalize direction and add some randomness
        if np.any(direction != 0):
            direction = direction / np.linalg.norm(direction)
        direction += np.random.normal(0, 0.1, 2)  # Add noise
        
        return np.clip(direction, -1, 1)  # Clip to valid action range
        
    def _log_progress(self, episode: int):
        """Log training progress."""
        # Get metrics
        metrics = self.metrics.get_statistics()
        
        # Log to tensorboard
        for metric, value in metrics.items():
            self.writer.add_scalar(metric, value, episode)
            
        # Log curriculum stage
        current_stage = self.curriculum_stages[self.current_stage]
        self.writer.add_scalar('curriculum/stage', self.current_stage, episode)
        self.writer.add_scalar('curriculum/difficulty', current_stage['difficulty'], episode)
        self.writer.add_scalar('curriculum/progress', self.stage_progress, episode)
        
        # Log adversarial training
        self.writer.add_scalar('adversarial/steps', self.adversarial_steps, episode)
        
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = os.path.join('hrl', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Ensure path doesn't contain directory separators
        filename = os.path.basename(path)
        if not filename.endswith('.pth'):
            filename += '.pth'
            
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'option_selector_state_dict': self.option_selector.state_dict(),
            'metrics': self.metrics.get_metrics(),
            'config': self.config,
            'current_stage': self.current_stage,
            'stage_progress': self.stage_progress,
            'adversarial_steps': self.adversarial_steps
        }
        
        # Save with pickle_protocol=4 for better compatibility
        torch.save(checkpoint, checkpoint_path, pickle_protocol=4)
        
        if self.debug_level >= 1:
            print(f"Checkpoint saved to {checkpoint_path}")
        
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        # Check if path is a filename only or a full path
        if os.path.dirname(path) == '':
            # Assume it's in the checkpoints directory
            checkpoint_path = os.path.join('hrl', 'checkpoints', path)
            if not os.path.exists(checkpoint_path) and not checkpoint_path.endswith('.pth'):
                checkpoint_path += '.pth'
        else:
            checkpoint_path = path
            
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint {checkpoint_path} not found.")
            return False
            
        try:
            # Set weights_only=False to handle PyTorch 2.6 changes
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.option_selector.load_state_dict(checkpoint['option_selector_state_dict'])
            self.metrics.metrics = checkpoint['metrics']
            self.current_stage = checkpoint['current_stage']
            self.stage_progress = checkpoint['stage_progress']
            self.adversarial_steps = checkpoint['adversarial_steps']
            
            print(f"Checkpoint loaded from {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
        
    def _log_metrics(self, episode: int):
        """Log training metrics."""
        # Get recent experiences
        if not hasattr(self, 'metrics'):
            print("Warning: Metrics tracker not initialized. Skipping logging.")
            return
            
        # Log basic metrics
        self.metrics.log_training_metric('episode', episode)
        
        # Initialize variables
        avg_reward = 0.0
        win_rate = 0.0
        recent_batch = []
        
        # Log policy metrics
        if hasattr(self.policy, 'buffer') and len(self.policy.buffer) > 0:
            recent_batch = self.policy.buffer.sample(min(100, len(self.policy.buffer)))
            avg_reward = sum(exp.reward for exp in recent_batch) / len(recent_batch)
            win_count = sum(1 for exp in recent_batch if exp.done and exp.reward > 0)
            done_count = sum(1 for exp in recent_batch if exp.done)
            win_rate = win_count / max(1, done_count)  # Avoid division by zero
            
            self.metrics.log_training_metric('avg_reward', avg_reward)
            self.metrics.log_game_metric('win_rate', win_rate)
            
            # Calculate additional metrics
            action_magnitude = np.mean([np.linalg.norm(exp.action) for exp in recent_batch])
            unique_options = len(set(exp.option for exp in recent_batch))
            # Handle empty batches properly
            max_reward = float('-inf')
            min_reward = float('inf')
            if recent_batch:
                max_reward = max(exp.reward for exp in recent_batch)
                min_reward = min(exp.reward for exp in recent_batch)
            
            # Log these additional metrics
            self.metrics.log_training_metric('action_magnitude', action_magnitude)
            self.metrics.log_training_metric('unique_options', unique_options)
            self.metrics.log_training_metric('max_reward', max_reward)
            self.metrics.log_training_metric('min_reward', min_reward)
            
        # Log option usage
        if recent_batch:
            option_counts = {}
            for exp in recent_batch:
                opt = exp.option
                option_counts[opt] = option_counts.get(opt, 0) + 1
                
            for opt, count in option_counts.items():
                usage_percent = count / len(recent_batch) * 100
                self.metrics.log_option_metric(f'usage_{opt}', count / len(recent_batch))
                # Also log this to console for visibility
                print(f"  Option {opt}: {usage_percent:.1f}% usage")
            
        # Log curriculum progress
        self.metrics.log_training_metric('curriculum_stage', self.current_stage)
        self.metrics.log_training_metric('stage_progress', self.stage_progress)
        
        # Print summary with enhanced details
        print(f"Episode {episode}: Reward={avg_reward:.2f} (min={min_reward:.2f}, max={max_reward:.2f}), Win Rate={win_rate:.2f}, Stage={self.current_stage}")
        print(f"  Action magnitude: {action_magnitude:.2f}, Unique options used: {unique_options}")
        
        # Save metrics
        if episode % (self.log_interval * 5) == 0:
            self.metrics.save_metrics(f"{self.log_dir}/metrics_episode_{episode}.json") 