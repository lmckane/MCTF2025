import numpy as np
from hrl.test.test_hrl import HRLEvaluator
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
from hrl.utils.option_initiation import OptionInitiation
from hrl.utils.option_transition import OptionTransition
from hrl.utils.option_reward import OptionReward
from hrl.utils.option_learning import OptionLearner
from hrl.utils.option_optimization import OptionOptimizer
from hrl.utils.option_evaluation import OptionEvaluator
from hrl.utils.option_selection import OptionSelector
from hrl.utils.option_execution import OptionExecutor
from hrl.utils.option_monitoring import OptionMonitor
from hrl.utils.option_adaptation import OptionAdapter
from hrl.utils.option_coordination import OptionCoordinator
from hrl.utils.option_communication import OptionCommunicator
from hrl.utils.option_planning import OptionPlanner
from hrl.utils.option_memory import OptionMemory
from hrl.utils.option_attention import OptionAttention
from hrl.utils.option_curiosity import OptionCuriosity
from hrl.utils.option_exploration import OptionExplorer
from hrl.utils.option_meta_learning import OptionMetaLearner
from hrl.utils.option_transfer import OptionTransfer
from hrl.utils.option_robustness import OptionRobustness
from hrl.utils.option_safety import OptionSafety
from hrl.utils.option_efficiency import OptionEfficiency
from hrl.utils.option_scalability import OptionScalability
from hrl.utils.option_interpretability import OptionInterpretability
from hrl.utils.option_debugging import OptionDebugger
from hrl.utils.option_logging import OptionLogger
from hrl.utils.option_visualization import OptionVisualizer
from hrl.utils.option_analysis import OptionAnalyzer
from pyquaticus.envs.rllib_pettingzoo_wrapper import ParallelPettingZooWrapper
from pyquaticus import pyquaticus_v0

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
    env = ParallelPettingZooWrapper(pyquaticus_v0.PyQuaticusEnv(
        render_mode="human" if config["render"] else None,
        team_size=config["env_config"]["team_size"],
        config_dict=config["env_config"]["config_dict"]
    ))
    
    # Get observation and action spaces
    obs_space = env.observation_space["agent_0"]
    act_space = env.action_space["agent_0"]
    
    # Add spaces to policy config
    config["policy_config"] = {
        "observation_space": obs_space,
        "action_space": act_space,
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
    option_initiation = OptionInitiation(config)
    option_transition = OptionTransition(config)
    option_reward = OptionReward(config)
    option_learner = OptionLearner(config)
    option_optimizer = OptionOptimizer(config)
    option_evaluator = OptionEvaluator(config)
    option_selector = OptionSelector(config)
    option_executor = OptionExecutor(config)
    option_monitor = OptionMonitor(config)
    option_adapter = OptionAdapter(config)
    option_coordinator = OptionCoordinator(config)
    option_communicator = OptionCommunicator(config)
    option_planner = OptionPlanner(config)
    option_memory = OptionMemory(config)
    option_attention = OptionAttention(config)
    option_curiosity = OptionCuriosity(config)
    option_explorer = OptionExplorer(config)
    option_meta_learner = OptionMetaLearner(config)
    option_transfer = OptionTransfer(config)
    option_robustness = OptionRobustness(config)
    option_safety = OptionSafety(config)
    option_efficiency = OptionEfficiency(config)
    option_scalability = OptionScalability(config)
    option_interpretability = OptionInterpretability(config)
    option_debugger = OptionDebugger(config)
    option_logger = OptionLogger(config)
    option_visualizer = OptionVisualizer(config)
    option_analyzer = OptionAnalyzer(config)
    
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
    
    # Analyze option performance
    print("\nAnalyzing option performance...")
    option_analyzer.analyze_performance(evaluator.metrics)
    
    # Log results
    option_logger.log_metrics(evaluator.metrics)
    
    # Debug any issues
    option_debugger.check_for_issues(evaluator.metrics)

if __name__ == "__main__":
    main() 