"""
Evaluate an agent that has been trained with the DQN algorithm to play the game Pong for Atari 2600.
"""

from dqn.dqn_agent import Agent
from dqn.pong_environment import PongFromFeatures

import numpy as np
import torch

NUMBER_OF_EVALUATION_EPISODES = 10
MAXIMUM_NUMBER_OF_STEPS_PER_EPISODE = 50000
SHOW_RENDERING = True

environment = PongFromFeatures(SHOW_RENDERING)
environment.seed(42)
print("State shape: ", environment.observation_space.shape)
print("Number of actions: ", environment.action_space.n)

agent = Agent(
    state_size=environment.observation_space.shape[0],
    action_size=environment.action_space.n,
    seed=0,
)

agent.qnetwork_local.load_state_dict(
    torch.load("./dqn/saved_models/dqm_episode_9000.pth")
)

rewards_per_episode_list = []

for i in range(NUMBER_OF_EVALUATION_EPISODES):
    print("Simulating episode: ", i)
    state = environment.reset()
    total_episode_reward = 0
    for j in range(MAXIMUM_NUMBER_OF_STEPS_PER_EPISODE):
        action = agent.act(state)
        environment.render()
        state, reward, done, _ = environment.step(action)
        total_episode_reward += reward
        if done:
            break
    rewards_per_episode_list.append(total_episode_reward)

environment.close()
rewards_per_episode_array = np.array(rewards_per_episode_list)

print("Average score: ", np.mean(rewards_per_episode_array))
print("Standard deviation of score: ", np.std(rewards_per_episode_array))
