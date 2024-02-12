"""Utilities to train an Agent to play the Pong game."""

import numpy as np


def mapAction(input):
    """
    Maps action according to policy definition to action according to gymnasium environment.

    @param input Action according to policy definition
    @return Action according to gymnasium environment.
    """
    if input == 0:
        return 1
    if input == 1:
        return 4
    if input == 2:
        return 5
    assert False, "unexpected action"


def preProcess(image):
    """
    Extract features from Pong image.

    @param image Image from the game Pong
    @return Extracted features
    """
    sliced_image_second_channel_channel = image[34:-16, :, 1]

    agent_paddle_channel_value = 186
    ball_channel_value = 236
    opponent_paddle_channel_value = 130

    vertical_slice_over_agent_paddle = sliced_image_second_channel_channel[:, 141]
    vertical_slice_over_opponent_paddle = sliced_image_second_channel_channel[:, 17]

    agent_paddle_coordinate = -10
    coordinates_agent_paddle = np.argwhere(
        vertical_slice_over_agent_paddle == agent_paddle_channel_value
    )
    if coordinates_agent_paddle.shape[0] > 0:
        agent_paddle_coordinate = np.mean(coordinates_agent_paddle)

    opponent_paddle_coordinate = -10
    coordiantes_opponent_paddle = np.argwhere(
        vertical_slice_over_opponent_paddle == opponent_paddle_channel_value
    )
    if coordiantes_opponent_paddle.shape[0] > 0:
        opponent_paddle_coordinate = np.mean(coordiantes_opponent_paddle)

    ball_coordinate = np.array([-10, -10])
    coordinates_ball = np.argwhere(
        sliced_image_second_channel_channel == ball_channel_value
    )
    if coordinates_ball.shape[0] > 0:
        ball_coordinate = np.mean(coordinates_ball, axis=0)

    return (
        sliced_image_second_channel_channel,
        agent_paddle_coordinate,
        opponent_paddle_coordinate,
        ball_coordinate,
    )


def aggregateFeatures(
    agent_paddle_coordinate, opponent_paddle_coordinate, ball_coordinate
):
    """
    Aggregate features to numpy array.
    """
    return np.array(
        [
            agent_paddle_coordinate,
            opponent_paddle_coordinate,
            ball_coordinate[0],
            ball_coordinate[1],
        ]
    )
