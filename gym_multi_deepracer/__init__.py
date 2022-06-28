from .multi_deepracer import MultiDeepRacer
import gym

gym.envs.register(
    id="MultiDeepRacer-v0",
    entry_point="gym_multi_deepracer:MultiDeepRacer",
    max_episode_steps=1000,
    reward_threshold=900,
)
