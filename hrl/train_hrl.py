# train_hrl.py — Enhanced Training Loop for MCTF HRL Agent with Class Structure

import os
import numpy as np
import ray
from ray.tune.logger import pretty_print
from hrl.utils.reward_shaping import RewardShaper
from hrl.utils.state_processor import StateProcessor
from hrl.utils.experience_buffer import ExperienceBuffer
from hrl.utils.option_selection import OptionSelector
from hrl.utils.option_execution import OptionExecutor
from hrl.utils.option_learning import OptionLearner
from hrl.utils.option_termination import OptionTermination

# === Config ===
class Config:
    max_episodes = 1000
    max_steps_per_episode = 500
    eval_interval = 50
    save_interval = 100
    gamma = 0.99
    lr = 1e-4
    device = "cpu"
    log_dir = "logs/"
    render = True
    env_config = {
        "env_bounds": [160.0, 80.0],
        "agent_radius": 2.0,
        "flag_radius": 2.0,
        "catch_radius": 10.0,
    }

class HRLTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.reward_shaper = RewardShaper(config)
        self.state_processor = StateProcessor(config)
        self.experience_buffer = ExperienceBuffer(config)
        self.option_selector = OptionSelector(config)
        self.option_executor = OptionExecutor(config)
        self.option_learner = OptionLearner(config)
        self.option_termination = OptionTermination(config)

        from pyquaticus.envs.mctf_env import parallel_env
        self.env = parallel_env(render_mode="human" if config.render else None)

    def train(self):
        for episode in range(self.config.max_episodes):
            obs = self.env.reset()
            state = self.state_processor.process(obs)

            done = {agent: False for agent in obs}
            episode_reward = {agent: 0 for agent in obs}
            current_option = {agent: self.option_selector.select(state[agent]) for agent in obs}

            for step in range(self.config.max_steps_per_episode):
                actions = {}
                for agent in obs:
                    if self.option_termination.should_terminate(agent, state[agent], current_option[agent]):
                        current_option[agent] = self.option_selector.select(state[agent])
                    actions[agent] = self.option_executor.execute(current_option[agent], state[agent])

                next_obs, reward, terminated, truncated, info = self.env.step(actions)
                next_state = self.state_processor.process(next_obs)
                shaped_reward = self.reward_shaper.shape(state, actions, reward, next_state)
                self.experience_buffer.store(obs, actions, shaped_reward, next_obs, current_option)

                for agent in obs:
                    episode_reward[agent] += shaped_reward[agent]

                obs = next_obs
                state = next_state

                if all(terminated.values()):
                    break

            self.option_learner.learn(self.experience_buffer)

            if episode % self.config.eval_interval == 0:
                print(f"Episode {episode + 1}: Reward Summary = {episode_reward}")

            if episode % self.config.save_interval == 0:
                save_path = os.path.join(self.config.log_dir, f"checkpoint_{episode + 1}.pt")
                self.option_learner.save(save_path)

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    config = Config()
    trainer = HRLTrainer(config)
    trainer.train()
