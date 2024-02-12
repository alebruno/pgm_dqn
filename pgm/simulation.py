"""
Abstraction of the environment.
"""

from collections import deque
from pgm.neural_network import Policy
from common.utilities import mapAction, preProcess, aggregateFeatures

import gymnasium as gym
import numpy as np


class Simulation:
    """
    Abstraction of an agent placed into an environment.
    """

    RENDER_MODE_HUMAN = "human"
    RENDER_MODE_NONE = None

    def __init__(self, policy=Policy(), enable_rendering=False):
        """
        Initialize the simulation.

        @param policy The policy of the agent placed in the environment.
        @param enable_rendering Flag to enable rendering.
        """
        self.env = gym.make(
            "ALE/Pong-v5",
            render_mode=(
                self.RENDER_MODE_HUMAN if enable_rendering else self.RENDER_MODE_NONE
            ),
        )
        self.multi_frame_observation = deque([-10] * 12, maxlen=12)
        self.policy = policy

    def simulate_episode(self, max_steps=5000):
        """
        Simulate a single episode.

        @param max_steps Maximum number of steps to be simulated.
        @return The reward list and the log probability of each action,
        including the derivatives w.r.t. the parameters of the model.
        """
        self.env.reset()
        raw_reward_list = []
        log_prob_list = []
        for i in range(max_steps):
            action, log_prob = self.policy.get_action(
                np.array(self.multi_frame_observation)
            )

            color_image, reward, terminated, truncated, info = self.env.step(
                mapAction(action)
            )

            (
                grayscale_image,
                agent_paddle_coordinate,
                opponent_paddle_coordinate,
                ball_coordinate,
            ) = preProcess(color_image)

            instantaneous_observation = aggregateFeatures(
                agent_paddle_coordinate, opponent_paddle_coordinate, ball_coordinate
            )
            for element in instantaneous_observation:
                self.multi_frame_observation.append(element)

            raw_reward_list.append(reward)
            log_prob_list.append(log_prob)

            if terminated or truncated:
                break

        return raw_reward_list, log_prob_list
