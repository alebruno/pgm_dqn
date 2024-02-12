"""
Implementation of the environment to play Pong.
"""

import gymnasium as gym
from gymnasium import spaces
from collections import deque
from common.utilities import mapAction, preProcess, aggregateFeatures

import numpy as np


class PongFromFeatures(gym.Env):
    """
    Environment of the game Pong that produces an observation based on hand-crafted features.
    """

    RENDER_MODE_HUMAN = "human"
    RENDER_MODE_NONE = None
    INITIAL_MULTI_FRAME_OBSERVATION = deque([-10] * 12, maxlen=12)

    def __init__(self, enable_rendering=False):
        """
        Initializes the environment.

        @parameter enable_rendering Enable rendering.
        """
        self.enable_rendering = enable_rendering
        self.env = gym.make(
            "ALE/Pong-v5",
            render_mode=(
                self.RENDER_MODE_HUMAN if enable_rendering else self.RENDER_MODE_NONE
            ),
        )
        self.multi_frame_observation = self.INITIAL_MULTI_FRAME_OBSERVATION

        # observation_space variable set in a way that is not compliant to gymnasium API.
        class MockObservationSpace:
            shape = (12,)

        self.observation_space = MockObservationSpace()

        self.action_space = spaces.Discrete(3)

    def reset(self):
        """
        Resets the environment.

        @return Extracted features and default values for the reward, the termination flag and the truncation flag.
        """
        self.multi_frame_observation = self.INITIAL_MULTI_FRAME_OBSERVATION
        color_image, _ = self.env.reset()
        return self.extractFeatures(color_image)[0]

    def step(self, action):
        """
        Does one step of the environment.

        @param action The action being taken.
        @return Extracted features, the reward, the termination flag and the truncation flag.
        """
        color_image, reward, terminated, truncated, _ = self.env.step(mapAction(action))
        return self.extractFeatures(color_image, reward, terminated, truncated)

    def extractFeatures(self, color_image, reward=0, terminated=False, truncated=False):
        """
        Extracts features from the color image and propagates the reward, the terminated and the truncated flags.

        @param color_image The color image of the Pong game.
        @param reward The value of the reward.
        @param terminated The terminated flag.
        @param truncated The truncated flag.
        @return Extracted features, the reward, the termination flag and the truncation flag.
        """
        (
            _,
            agent_paddle_coordinate,
            opponent_paddle_coordinate,
            ball_coordinate,
        ) = preProcess(color_image)

        instantaneous_observation = aggregateFeatures(
            agent_paddle_coordinate, opponent_paddle_coordinate, ball_coordinate
        )
        for element in instantaneous_observation:
            self.multi_frame_observation.append(element)

        done = True if terminated or truncated else False

        # Deviation from gymnasium API: truncated is not used and always set to false.
        output_terminated = done
        output_truncated = False

        return (
            np.array(self.multi_frame_observation),
            reward,
            output_terminated,
            output_truncated,
        )

    def seed(self, value):
        """
        Sets the seed.

        @param seed Value of the seed.
        """
        self.env.seed(value)

    def close(self):
        """
        Closes the environment.
        """
        self.env.close()
        self.multi_frame_observation = self.INITIAL_MULTI_FRAME_OBSERVATION

    def render(self):
        """
        Visualizes the environment.
        """
        if self.enable_rendering:
            self.env.render()
