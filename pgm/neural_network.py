"""
Implementation of the policy using a neural network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

torch.manual_seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    """
    Neural network implementing the policy.
    """

    def __init__(
        self, observation_size=12, hidden_size_1=64, hidden_size_2=64, action_size=3
    ):
        """
        Initialize the neural network implementing the policy.

        @param observation_size Size of the obsevation vector.
        @param hidden_size_1 Number of neurons in the first hidden layer.
        @param hidden_size_2 Number of neurons in the second hidden layer.
        @param action_size Number of available actions.

        """
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(observation_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, action_size)

    def forward(self, observation):
        """
        Calculate the probability of choosing each available action.

        @param observation The observation vector.
        @return Probabilities of choosing each available action.
        """
        x = observation / 100.0
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return nn.functional.softmax(x, dim=1)

    def get_action(self, observation):
        """
        Calculate the log probabilities and sample one action from the probability distribution.

        @param observation The observation vector.
        @return Sampled action and log probability for each available action.
        """
        observation_pytorch = (
            torch.from_numpy(observation).float().unsqueeze(0).to("cpu")
        )
        probs = self.forward(observation_pytorch).cpu()
        categorical_probability_distribution = Categorical(probs)
        action = categorical_probability_distribution.sample()
        return action.item(), categorical_probability_distribution.log_prob(action)
