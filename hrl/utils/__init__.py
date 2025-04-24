"""
Utility modules for the HRL framework.

This package contains various utility modules used throughout the HRL framework,
including state processing, option selection, team coordination, and metrics tracking.

Important Notes:
- The canonical implementation of OptionSelector is in option_selector.py
- Previous redundant implementations in option_selection.py and policies/option_selector.py
  have been removed to maintain clarity and avoid confusion
"""

from hrl.utils.option_selector import OptionSelector
from hrl.utils.state_processor import StateProcessor
from hrl.utils.team_coordinator import TeamCoordinator, AgentRole
from hrl.utils.metrics import MetricsTracker
from hrl.utils.reward_shaping import RewardShaper
from hrl.utils.experience_buffer import ExperienceBuffer
from hrl.utils.option_execution import OptionExecutor
from hrl.utils.option_learning import OptionLearner
from hrl.utils.option_termination import OptionTermination
from hrl.utils.opponent_modeler import OpponentModeler
from hrl.utils.experience import Experience, ReplayBuffer
from hrl.utils.option_monitoring import OptionMonitor

__all__ = [
    'OptionSelector',
    'StateProcessor',
    'TeamCoordinator',
    'AgentRole',
    'MetricsTracker',
    'RewardShaper',
    'ExperienceBuffer',
    'OptionExecutor',
    'OptionLearner',
    'OptionTermination',
    'OpponentModeler',
    'Experience',
    'ReplayBuffer',
    'OptionMonitor'
] 