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
from hrl.environment.pyquaticus_wrapper import PyquaticusWrapper
from hrl.utils.team_coordinator import TeamCoordinator
from hrl.utils.opponent_modeler import OpponentModeler

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
                - self_play_config: Self-play parameters
                - opponent_config: Opponent strategy parameters
        """
        self.config = config
        self.env_config = config['env_config']
        self.policy_config = config['policy_config']
        self.training_config = config['training_config']
        self.curriculum_config = config['curriculum_config']
        self.self_play_config = config.get('self_play_config', {'enabled': False})
        self.opponent_config = config.get('opponent_config', {})
        
        # Initialize environment
        self.env = GameEnvironment(self.env_config)
        
        # Initialize components
        self.state_processor = StateProcessor(self.policy_config)
        self.option_selector = OptionSelector(self.policy_config)
        
        # Ensure policy_config has the required parameters
        self.policy_config['action_size'] = 2  # 2D movement
        
        self.policy = HierarchicalPolicy(
            config=self.policy_config
        )
        
        # Create team coordinator
        self.team_coordinator = TeamCoordinator(self.curriculum_config['stages'][0])
        
        # Create opponent modeler
        self.opponent_modeler = OpponentModeler({
            'debug_level': self.training_config.get('debug_level', 1),
            'num_opponents': self.opponent_config.get('num_opponents', 3)
        })
        
        # Initialize metrics
        self.metrics = MetricsTracker(self.training_config.get('metrics_config', {}))
        
        # Training parameters
        self.num_episodes = self.training_config.get('num_episodes', 10000)
        self.max_steps = self.training_config.get('max_steps', 1000)
        self.batch_size = self.training_config.get('batch_size', 64)
        self.gamma = self.policy_config.get('gamma', 0.99)
        self.lambda_ = self.policy_config.get('lambda_', 0.95)
        self.entropy_coef = self.policy_config.get('entropy_coef', 0.01)
        self.learning_rate = self.policy_config.get('learning_rate', 0.001)
        self.log_interval = self.training_config.get('log_interval', 100)
        self.checkpoint_interval = self.training_config.get('checkpoint_interval', 1000)
        self.render = self.training_config.get('render', False)
        self.debug_level = self.training_config.get('debug_level', 1)
        
        # Checkpoint preservation settings
        self.max_checkpoints = self.training_config.get('max_checkpoints', 5)
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_wins = []
        self.episode_scores = []
        self.current_eval_results = {'win_rate': 0.0, 'avg_score': 0.0}
        
        # Curriculum parameters
        self.curriculum_enabled = self.curriculum_config.get('enabled', True)
        self.curriculum_stages = self.curriculum_config.get('stages', [])
        self.current_stage = 0
        self.stage_progress = 0.0
        self.progression_metric = self.curriculum_config.get('progression_metric', 'win_rate')
        self.progression_threshold = self.curriculum_config.get('progression_threshold', 0.6)
        self.min_episodes_per_stage = self.curriculum_config.get('min_episodes_per_stage', 500)
        
        # Self-play parameters
        self.self_play_enabled = self.self_play_config.get('enabled', False)
        self.self_play_start_episode = self.self_play_config.get('start_episode', 1000)
        self.self_play_frequency = self.self_play_config.get('frequency', 0.3)
        self.policy_bank_size = self.self_play_config.get('policy_bank_size', 5)
        self.policy_bank_update_freq = self.self_play_config.get('policy_bank_update_freq', 500)
        self.policy_bank = []  # Store past versions of the policy
        
        # Early stopping
        self.eval_interval = self.training_config.get('eval_interval', 100)
        self.early_stopping_patience = self.training_config.get('early_stopping_patience', 20)
        self.best_performance = 0.0
        self.patience_counter = 0
        self.early_stopping_triggered = False
        
        # Create log directory
        self.log_dir = os.path.join(
            self.training_config.get('log_dir', 'logs'),
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # Create checkpoint directory
        self.checkpoint_dir = self.training_config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Adversarial training parameters
        self.adversarial_steps = 0
        self.adversarial_update_freq = self.training_config.get('adversarial_update_freq', 100)
        self.adversarial_ratio = self.training_config.get('adversarial_ratio', 0.5)
        
    def train(self):
        """Run training loop with curriculum learning and self-play."""
        print(f"\n{'='*80}")
        print(f"Starting training with {self.num_episodes} episodes")
        
        if self.curriculum_enabled:
            print(f"Curriculum learning enabled with {len(self.curriculum_stages)} stages")
            for i, stage in enumerate(self.curriculum_stages):
                print(f"  Stage {i+1}: {stage['name']} (difficulty={stage['difficulty']})")
                strategies = stage.get('opponent_strategies', {})
                if strategies:
                    print(f"    Opponent strategies: {', '.join([f'{k}={v:.1%}' for k, v in strategies.items() if v > 0])}")
        else:
            print("Curriculum learning disabled - using fixed difficulty")
        
        if self.self_play_enabled:
            print(f"Self-play enabled (starts at episode {self.self_play_start_episode})")
            print(f"Policy bank size: {self.policy_bank_size}")
            print(f"Policy bank update frequency: every {self.policy_bank_update_freq} episodes")
        else:
            print("Self-play disabled")
        
        print(f"{'='*80}")
        
        print("Setting up training environment...")
        
        total_episodes = 0
        start_time = time.time()
        
        # Calculate episodes per stage if using curriculum
        episodes_per_stage = []
        if self.curriculum_enabled:
            for stage in self.curriculum_stages:
                episodes_per_stage.append(int(self.num_episodes * stage['duration']))
        else:
            # If curriculum disabled, use single stage
            episodes_per_stage = [self.num_episodes]
        
        # Training loop across stages
        for stage_idx in range(len(episodes_per_stage)):
            if self.early_stopping_triggered:
                print("\nEarly stopping triggered - ending training")
                break
            
            print(f"\n{'-'*60}")
            if self.curriculum_enabled:
                print(f"Starting training stage {stage_idx + 1}: {self.curriculum_stages[stage_idx]['name']}")
                print(f"Difficulty: {self.curriculum_stages[stage_idx]['difficulty']}")
            else:
                print(f"Starting training (fixed difficulty={self.env_config['difficulty']})")
            print(f"{'-'*60}")
            
            # Update curriculum stage
            self.current_stage = stage_idx
            self.stage_progress = 0.0
            
            stage_episodes = episodes_per_stage[stage_idx]
            
            # Training loop within each stage
            for episode in range(stage_episodes):
                if self.early_stopping_triggered:
                    break
                    
                total_episodes += 1
                
                # Determine if this episode uses self-play
                use_self_play = (self.self_play_enabled and 
                               total_episodes > self.self_play_start_episode and
                               random.random() < self.self_play_frequency)
                
                # Set up environment and opponents
                env_config = self._setup_episode_environment(use_self_play)
                env = PyquaticusWrapper(env_config)
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
                
                # Update policy bank for self-play
                if self.self_play_enabled and total_episodes > 0 and total_episodes % self.policy_bank_update_freq == 0:
                    # Save a checkpoint specifically for the policy bank
                    policy_bank_name = f"policy_bank_ep{total_episodes}"
                    policy_bank_path = self.save_checkpoint(policy_bank_name, preserve_history=False)
                    
                    # Add to policy bank and limit size
                    self.policy_bank.append(policy_bank_path)
                    if len(self.policy_bank) > self.policy_bank_size:
                        # Remove oldest policy
                        self.policy_bank.pop(0)
                    
                    print(f"\nUpdated policy bank at episode {total_episodes} (size: {len(self.policy_bank)})")
                
                # Log metrics
                if episode % self.log_interval == 0 or episode == stage_episodes - 1:
                    self._log_detailed_metrics(episode, total_episodes, stage_idx, episode_time)
                
                # Print progress
                if episode % max(1, stage_episodes // 100) == 0:
                    self._print_progress(episode, stage_episodes, total_episodes, stage_idx)
                
                # Save checkpoint
                if episode % self.checkpoint_interval == 0:
                    if isinstance(stage, dict):
                        stage_name = stage.get('name', 'stage')
                        self.save_checkpoint(f"{stage_name}_episode_{episode}")
                    else:
                        self.save_checkpoint(f"stage_{stage_idx}_episode_{episode}")
                    
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
        
    def _print_progress(self, episode, episodes_in_stage, total_episodes, stage_idx):
        """Print training progress in a concise format."""
        # Calculate recent statistics
        recent_rewards = self.episode_rewards[-100:]
        recent_wins = self.episode_wins[-100:]
        recent_scores = self.episode_scores[-100:] if self.episode_scores else [0]
        
        # Print progress bar and stats
        progress = episode / episodes_in_stage * 100
        
        # Handle stage parameter correctly whether it's a dict or an int
        if isinstance(stage_idx, dict):
            stage_name = stage_idx.get('name', 'Unknown')
            stage_idx = self.curriculum_stages.index(stage_idx) if stage_idx in self.curriculum_stages else 0
            stage_display = f"[Stage {stage_idx+1}/{len(self.curriculum_stages)}] {stage_name}"
        else:  # stage_idx is an int
            stage_name = self.curriculum_stages[stage_idx]['name'] if 0 <= stage_idx < len(self.curriculum_stages) else "Unknown"
            stage_display = f"[Stage {stage_idx+1}/{len(self.curriculum_stages)}] {stage_name}"
        
        print(f"\r{stage_display} "
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
        
    def _log_detailed_metrics(self, episode, total_episodes, stage_idx, episode_time):
        """Log detailed metrics and print summary."""
        # Get recent metrics
        recent_rewards = self.episode_rewards[-100:]
        recent_wins = self.episode_wins[-100:]
        recent_lengths = self.episode_lengths[-100:]
        recent_scores = self.episode_scores[-100:] if self.episode_scores else [0]
        
        # Log to metrics tracker
        self.metrics.log_training_metric('episode', total_episodes)
        self.metrics.log_training_metric('stage', stage_idx)
        self.metrics.log_game_metric('win_rate', np.mean(recent_wins))
        self.metrics.log_training_metric('avg_reward', np.mean(recent_rewards))
        self.metrics.log_training_metric('episode_length', np.mean(recent_lengths))
        self.metrics.log_game_metric('avg_score', np.mean(recent_scores))
        
        # Print detailed summary
        if self.debug_level >= 1:
            print(f"\n{'-'*80}")
            # Handle stage parameter correctly whether it's a dict or an int
            if isinstance(stage_idx, dict):
                stage_name = stage_idx.get('name', 'Unknown')
                print(f"Episode {total_episodes} Summary (Stage: {stage_name})")
            else:  # stage_idx is an int
                stage_name = self.curriculum_stages[stage_idx]['name']
                print(f"Episode {total_episodes} Summary (Stage {stage_idx+1}: {stage_name})")
            
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
        
        print("Creating environment with PyquaticusWrapper...")
        
        # Adjust environment parameters based on difficulty
        env_config = self.env_config.copy()
        env_config['difficulty'] = difficulty
        env_config['debug_level'] = self.debug_level  # Pass debug level to environment
        
        # Adjust parameters based on difficulty
        env_config['num_agents'] = 3  # Fixed for Pyquaticus 3v3
        env_config['tag_radius'] = 5 - difficulty * 2  # Smaller tag radius at higher difficulty
        env_config['capture_radius'] = 10 - difficulty * 3  # Smaller capture radius at higher difficulty
        env_config['render'] = self.render  # Pass render flag
        
        print(f"PyquaticusWrapper config: {env_config}")
        
        # Create Pyquaticus environment
        try:
            env = PyquaticusWrapper(env_config)
            print("PyquaticusWrapper created successfully!")
            return env
        except Exception as e:
            print(f"ERROR creating PyquaticusWrapper: {e}")
            import traceback
            traceback.print_exc()
            raise
        
    def _run_episode(self, env: PyquaticusWrapper) -> Dict[str, Any]:
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
        
    def save_checkpoint(self, path: str, preserve_history=True):
        """
        Save training checkpoint.
        
        Args:
            path: Base path/name for the checkpoint
            preserve_history: Whether to preserve this as a versioned checkpoint
            
        Returns:
            str: Path to the saved checkpoint (latest version)
        """
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = os.path.join('hrl', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Ensure path doesn't contain directory separators
        filename = os.path.basename(path)
        
        # Replace any dict in the path with a string representation
        if '{' in filename:
            # This is a dirty fix - replace the dict with a simpler string
            filename = filename.replace(filename[filename.find('{'):filename.rfind('}')+1], "current_stage")
        
        # Get current timestamp for versioning
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create a version-specific filename if preserving history
        if preserve_history:
            version_filename = f"{filename}_v{timestamp}.pth"
            checkpoint_version_path = os.path.join(checkpoint_dir, version_filename)
        
        # Always create the latest version filename
        if not filename.endswith('.pth'):
            filename += '.pth'
        checkpoint_latest_path = os.path.join(checkpoint_dir, filename)
        
        # Prepare checkpoint data
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'option_selector_state_dict': self.option_selector.state_dict(),
            'metrics': self.metrics.get_metrics(),
            'config': self.config,
            'current_stage': self.current_stage,
            'stage_progress': self.stage_progress,
            'adversarial_steps': self.adversarial_steps,
            'version': timestamp,
            'episode_rewards': self.episode_rewards[-100:],  # Save last 100 rewards
            'episode_wins': self.episode_wins[-100:],        # Save last 100 win records
            'episode_scores': self.episode_scores[-100:]     # Save last 100 scores
        }
        
        # Save versioned checkpoint if preserving history
        if preserve_history:
            torch.save(checkpoint, checkpoint_version_path, pickle_protocol=4)
            if self.debug_level >= 1:
                print(f"Versioned checkpoint saved to {checkpoint_version_path}")
        
        # Always save latest checkpoint
        torch.save(checkpoint, checkpoint_latest_path, pickle_protocol=4)
        if self.debug_level >= 1:
            print(f"Latest checkpoint saved to {checkpoint_latest_path}")
            
        # Clean up old versions if needed
        if preserve_history and hasattr(self, 'max_checkpoints'):
            self._cleanup_old_checkpoints(filename, checkpoint_dir)
            
        # Return the path to the latest checkpoint
        return checkpoint_latest_path
    
    def _cleanup_old_checkpoints(self, base_filename, checkpoint_dir):
        """
        Clean up old checkpoint versions to maintain a maximum number of saved versions.
        
        Args:
            base_filename: Base name of the checkpoint file
            checkpoint_dir: Directory containing checkpoints
        """
        # Get all versioned checkpoints for this base filename
        base_name = base_filename.replace('.pth', '')
        pattern = f"{base_name}_v*.pth"
        
        # List all matching files
        import glob
        files = glob.glob(os.path.join(checkpoint_dir, pattern))
        
        # If we have more files than the max allowed, delete the oldest ones
        if len(files) > self.max_checkpoints:
            # Sort files by creation time (oldest first)
            files.sort(key=os.path.getctime)
            
            # Remove oldest files to keep only max_checkpoints
            files_to_remove = files[:-self.max_checkpoints]
            for old_file in files_to_remove:
                if os.path.exists(old_file):
                    os.remove(old_file)
                    if self.debug_level >= 2:
                        print(f"Removed old checkpoint: {old_file}")
        
    def load_checkpoint(self, path: str):
        """
        Load training checkpoint.
        
        Args:
            path: Path to the checkpoint to load. Can be:
                - A full path
                - Just a filename (will look in checkpoints directory)
                - A model name without .pth extension
                - A versioned checkpoint name with 'v' followed by timestamp
                
        Returns:
            bool: True if checkpoint was loaded successfully, False otherwise
        """
        # Check if path is a filename only or a full path
        if os.path.dirname(path) == '':
            # Assume it's in the checkpoints directory
            checkpoint_path = os.path.join('hrl', 'checkpoints', path)
            if not os.path.exists(checkpoint_path) and not checkpoint_path.endswith('.pth'):
                checkpoint_path += '.pth'
                
            # If still not found, check if it's a request for a specific version
            if not os.path.exists(checkpoint_path) and '_v' not in path:
                # Try to find the latest version by timestamp
                import glob
                base_name = path.replace('.pth', '')
                pattern = os.path.join('hrl', 'checkpoints', f"{base_name}_v*.pth")
                versioned_files = glob.glob(pattern)
                
                if versioned_files:
                    # Sort by creation time, newest first
                    versioned_files.sort(key=os.path.getctime, reverse=True)
                    checkpoint_path = versioned_files[0]
                    print(f"Loading the most recent version: {os.path.basename(checkpoint_path)}")
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
            
            # Load extra data if available
            if 'episode_rewards' in checkpoint:
                self.episode_rewards = checkpoint['episode_rewards']
            if 'episode_wins' in checkpoint:
                self.episode_wins = checkpoint['episode_wins']
            if 'episode_scores' in checkpoint:
                self.episode_scores = checkpoint['episode_scores']
                
            # Print version information if available
            if 'version' in checkpoint:
                version_timestamp = checkpoint['version']
                print(f"Loaded checkpoint from {checkpoint_path} (version: {version_timestamp})")
            else:
                print(f"Loaded checkpoint from {checkpoint_path}")
                
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
        
    def _setup_episode_environment(self, use_self_play=False):
        """Set up environment configuration for an episode."""
        # Start with base environment config
        episode_env_config = self.env_config.copy()
        
        # Get current curriculum stage configuration
        if self.curriculum_enabled:
            stage = self.curriculum_stages[self.current_stage]
            episode_env_config['difficulty'] = stage['difficulty']
            
            # Apply opponent strategies from current stage
            if 'opponent_strategies' in stage:
                episode_env_config['opponent_strategies'] = stage['opponent_strategies']
        
        # If using self-play, configure opponent to use a policy from the bank
        if use_self_play and self.policy_bank:
            # Choose a random policy from the bank
            policy_idx = random.randint(0, len(self.policy_bank) - 1)
            
            # Get the policy path
            policy_path = self.policy_bank[policy_idx]
            
            # Set self-play config
            episode_env_config['self_play'] = True
            episode_env_config['opponent_policy'] = policy_path
            
            if self.debug_level >= 2:
                print(f"Using self-play with policy {os.path.basename(policy_path)}")
        else:
            episode_env_config['self_play'] = False
            
        # Apply any stage-specific environment modifications
        if self.curriculum_enabled and 'env_modifications' in stage:
            for key, value in stage['env_modifications'].items():
                episode_env_config[key] = value
                
        return episode_env_config 