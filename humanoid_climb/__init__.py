"""
Humanoid Climbing Reinforcement Learning Environment

A PyBullet-based gymnasium environment for training humanoid robots to climb walls
using reinforcement learning with stance-based progression.
"""

import gymnasium as gym
from gymnasium.envs.registration import register

# Register the HumanoidClimb environment
register(
    id='HumanoidClimb-v0',
    entry_point='humanoid_climb.env:HumanoidClimbEnv',
    max_episode_steps=1000,
)

__version__ = "0.1.0"
__author__ = "Humanoid Climb Team"
