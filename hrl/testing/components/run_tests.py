import numpy as np
from hrl.testing.components.test_hrl import HRLEvaluator
from hrl.utils.visualization import HRLVisualizer
from hrl.options import (
    CaptureFlagOption, GuardFlagOption, PatrolOption,
    AttackOption, DefendOption, SupportOption
)
from hrl.policies import (
    HierarchicalPolicy, OptionPolicy, MetaPolicy
)
from hrl.utils.reward_shaping import RewardShaper
from hrl.utils.state_processor import StateProcessor
from hrl.utils.experience_buffer import ExperienceBuffer
from hrl.utils.option_termination import OptionTermination
from hrl.utils.option_learning import OptionLearner
from hrl.utils.option_selector import OptionSelector
from hrl.utils.option_execution import OptionExecutor
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
from pyquaticus import pyquaticus_v0
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

def main():
    # Environment configuration
    env_bounds = [100, 100]  # Environment dimensions
    num_episodes = 100
    max_steps = 1000
    
    # Create configuration dictionary
    config = {
        "render": True,
        "env_config": {
            "team_size": 1,
            "config_dict": {
                "sim_speedup_factor": 16,
                "max_time": 1000,
                "lidar_obs": True,
                "num_lidar_rays": 100,
                "lidar_range": 20,
                "render_lidar_mode": "detection",
                "render_agent_ids": True,
                "render_traj_mode": "traj_history",
                "render_traj_freq": 50,
                "short_obs_hist_length": 4,
                "short_obs_hist_interval": 5,
                "long_obs_hist_length": 5,
                "long_obs_hist_interval": 20,
                "render_traj_cutoff": 100,
                "tag_on_oob": True
            }
        },
        "options": ["capture_flag", "guard_flag", "patrol", "attack", "defend", "support"],
    }
    
    # Initialize environment with wrapper
    env_creator = lambda con: pyquaticus_v0.PyQuaticusEnv(
        render_mode="human" if config["render"] else None,
        team_size=config["env_config"]["team_size"],
        config_dict=config["env_config"]["config_dict"]
    )
    
    # Register environment
    register_env("pyquaticus", lambda config: ParallelPettingZooWrapper(env_creator(config)))
    
    # Create wrapped environment
    env = ParallelPettingZooWrapper(env_creator(config))
    
    # Get observation and action spaces
    obs_space = env.observation_space["agent_0"]
    act_space = env.action_space["agent_0"]
    
    # Define policy mapping function
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "default_policy"
    
    # Define policies
    policies = {
        "default_policy": (None, obs_space, act_space, {})
    }
    
    # Add spaces and multi-agent config to policy config
    config["policy_config"] = {
        "train_batch_size": 4000,
        "num_sgd_iter": 10,
        "lr": 3e-4,
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "model": {
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "tanh"
        }
    }
    
    # Initialize PPO config
    ppo_config = PPOConfig().api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    ).environment(env="pyquaticus").env_runners(
        num_env_runners=1,
        num_cpus_per_env_runner=0.25
    )
    
    # Add multi-agent configuration
    ppo_config.multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["default_policy"]
    )
    
    # Build the algorithm
    algo = ppo_config.build_algo()
    
    # Initialize evaluator and visualizer
    evaluator = HRLEvaluator(config)
    visualizer = HRLVisualizer(env_bounds)
    
    # Define options
    options = {
        "capture_flag": CaptureFlagOption(config.get("capture_config", {})),
        "guard_flag": GuardFlagOption(config.get("guard_config", {})),
        "patrol": PatrolOption(config.get("patrol_config", {})),
        "attack": AttackOption(config.get("attack_config", {})),
        "defend": DefendOption(config.get("defend_config", {})),
        "support": SupportOption(config.get("support_config", {}))
    }
    
    # Initialize policies
    meta_policy = MetaPolicy(config.get("meta_config", {}))
    option_policies = {name: OptionPolicy(config.get("option_config", {})) for name in options.keys()}
    hierarchical_policy = HierarchicalPolicy(meta_policy, option_policies)
    
    # Initialize utility classes with config
    reward_shaper = RewardShaper(config)
    state_processor = StateProcessor(config)
    experience_buffer = ExperienceBuffer(config)
    option_termination = OptionTermination(config)
    option_learner = OptionLearner(config)
    option_selector = OptionSelector(config)
    option_executor = OptionExecutor(config)
        
    # Test individual options
    print("Testing individual options...")
    for option_name, option in options.items():
        print(f"\nTesting {option_name} option...")
        evaluator.test_option(option_name, num_episodes=10)
        evaluator.visualize_metrics()
    
    # Test full system
    print("\nTesting full HRL system...")
    evaluator.test_full_system(num_episodes=num_episodes)
    evaluator.visualize_metrics()
    
    # Record and save animation of a single episode
    print("\nRecording episode animation...")
    states, options_executed = evaluator.record_episode()
    visualizer.save_animation(states, options_executed, "episode_animation.gif")
    
    # Plot trajectory and heatmap
    print("\nGenerating trajectory and heatmap plots...")
    visualizer.plot_trajectory(states)
    visualizer.plot_heatmap()

if __name__ == "__main__":
    main() 