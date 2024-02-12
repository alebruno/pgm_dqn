"""
Train an agent that can play the game Pong for Atari 2600 using the REINFORCE algorithm, which is a Policy Gradient Method.
"""

from pgm.simulation import Simulation
from pgm.neural_network import Policy

import numpy as np
from torch.optim import Adam
import torch


def getFutureAmortizedRewards(raw_reward_list, gamma=0.97):
    """Calculates the list of amortized rewards using exonential decay.

    @param raw_reward_list  List of instantaneous rewards.
    @param gamma  Exponent of exponential decay.

    @return List of amortized rewards.
    """
    amortized_future_rewards_list = []
    elements = len(raw_reward_list)
    for i in range(elements):
        amortized_sum = 0
        factor = 1
        for j in range(i, elements):
            amortized_sum += factor * raw_reward_list[j]
            factor *= gamma
        amortized_future_rewards_list.append(amortized_sum)
    return amortized_future_rewards_list


NUMBER_OF_EPISODES_PER_BATCH = 10
NUMBER_OF_BATCHES = 30000

policy = Policy()

optimizer = Adam(policy.parameters(), lr=1e-3)
total_raw_reward_list = []
simulation = Simulation(policy)
for batch_counter in range(NUMBER_OF_BATCHES):
    print("Processing batch ", batch_counter)
    policy_loss = 0
    total_raw_reward = 0
    for episode_counter in range(NUMBER_OF_EPISODES_PER_BATCH):
        raw_reward_list, log_prob_list = simulation.simulate_episode()
        total_raw_reward += sum(raw_reward_list)
        amortized_future_rewards_list = getFutureAmortizedRewards(raw_reward_list)

        policy_loss_list = []
        for r, log_prob in zip(amortized_future_rewards_list, log_prob_list):
            policy_loss_list.append(-log_prob * r)

        policy_loss += torch.cat(policy_loss_list).sum()
    print("Average score: ", total_raw_reward)
    total_raw_reward_list.append(total_raw_reward)
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    if batch_counter % 100 == 0:
        torch.save(policy.state_dict(), "./pgm_model.pt")
        np.save("reward_training.npy", np.array(total_raw_reward_list))
