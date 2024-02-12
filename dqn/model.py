"""Approximation of the Q function."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Neural network to approximate the Q function."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=256):
        """
        Initialize parameters and build model.

        @param state_size (int): Dimension of each state
        @param action_size (int): Dimension of each action
        @param seed (int): Random seed
        @param fc1_units (int): Number of nodes in first hidden layer
        @param fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Estimates the Q function for a given state and all the available actions.

        @param state The observed state.
        @return Value of the Q function for all the available actions.
        """
        scaled_state = state / 100.0
        x = F.relu(self.fc1(scaled_state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
