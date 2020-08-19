import gym
import numpy as np
import matplotlib.pyplot as plt
from ddpg import DDPGAgent
from utils import *

env = gym.make("FetchReach-v1")

agent = DDPGAgent(env)
noise = OUNoise(env.action_space) # UO noise needs the action space to operate
batch_size = 128

# reward lists to plot training progress later
rewards = []
avg_rewards = []

num_episodes = 100

max_reward = -1000 ## arbitrarily large negative number

for episode in range (num_episodes):
	state = env.reset()["observation"] ## observation space in this case is a dictionary
	# state = env.reset()
	noise.reset()
	episode_reward = 0

	for step in range (500):
		action = agent.get_action(state)
		action = noise.get_action(action, step)
		next_state, reward, done, info = env.step(action)


		achieved_goal = next_state["achieved_goal"]
		desired_goal = next_state["desired_goal"]
		next_state = next_state["observation"]

		experience = (state, action, np.array([reward]), next_state, achieved_goal, desired_goal)
		agent.memory.store(experience)

		if agent.memory.__len__() > batch_size:
			agent.update(batch_size)

		state = next_state
		episode_reward += reward


		rewards.append(episode_reward)
		avg_rewards.append(np.mean(rewards[-10:]))

		# env.render()

	if (episode_reward > max_reward):
		max_reward = episode_reward
		agent.save_models()

	print("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))

	# print("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
