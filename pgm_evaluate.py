"""
Evaluate an agent that has been trained with the REINFORCE algorithm to play the game Pong for Atari 2600.
"""

from pgm.simulation import Simulation
from pgm.neural_network import Policy

import numpy as np
import torch

NUMBER_OF_EVALUATION_EPISODES = 10
SHOW_RENDERING = True

policy = Policy()
policy.load_state_dict(torch.load("./pgm/saved_models/pgm_batch_5000"))
policy.eval()

simulation = Simulation(policy, enable_rendering=SHOW_RENDERING)
total_raw_reward_per_episode = []
for episode_counter in range(NUMBER_OF_EVALUATION_EPISODES):
    raw_reward_list, _ = simulation.simulate_episode(max_steps=50000)
    sum_of_raw_rewards = sum(raw_reward_list)
    total_raw_reward_per_episode.append(sum_of_raw_rewards)
    print("Simulating episode ", episode_counter)
total_raw_reward_per_episode_array = np.array(total_raw_reward_per_episode)

print("Average score: ", np.mean(total_raw_reward_per_episode_array))
print("Standard deviation of score: ", np.std(total_raw_reward_per_episode_array))
