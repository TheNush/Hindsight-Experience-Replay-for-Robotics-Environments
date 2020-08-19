import numpy as np
import gym

from utils import *
from models import *
from ddpg import DDPGAgent
from strategy import *

import matplotlib.pyplot as plt

import time


env = gym.make('FetchReach-v1')

## initialize off-line learning algorithm - DDPG in this case
agent = DDPGAgent(env)

## initialize the strategy
strategy = Future(env)

## initialize OU Noise
noise = OUNoise(env.action_space)

NUM_OF_EPOCHS = 200
NUM_OF_EPISODES = 800
STEPS_PER_EPISODE = 50
K_FUTURE = 4
BATCH_SIZE = 128

episode_rewards = []

for epoch in range(NUM_OF_EPOCHS):
	for episode in range(NUM_OF_EPISODES):
		
		env_data = env.reset()
		
		standard_replay = []
		her_replay = []

		
		## extract data
		state = env_data["observation"]
		goal = env_data["desired_goal"]

		## logging rewards
		episode_reward = 0

		for step in range(STEPS_PER_EPISODE):
			## normalize state and goal
			# state = normalizer(state, 5.0)
			# goal = normalizer(goal, 5.0)

			## get action from behavioural policy
			action = agent.get_action(state, goal)
			if action is not None:
				action = noise.get_action(action, step)
			else:
				action = env.action_space.sample()

			time.sleep(0.002)
			next_state, reward, _, _ = env.step(action)
			
			env.render()

			episode_reward += reward

			## store transition - Standard Experience Replay
			state_rep = np.concatenate((state, goal), axis=0)
			next_state_rep = np.concatenate((next_state["observation"], goal), axis=0)
			standard_transition = [state_rep, action, reward, next_state_rep]
			# agent.memory.store(standard_transition)
			standard_replay.append(standard_transition)

			state = next_state["observation"]

			if agent.memory.__len__() > BATCH_SIZE:
				## perform one-step optimization on BATCH
				agent.update(batch_size = BATCH_SIZE)

		## the episode is now over
		## need to create normalized HER transitions using Strategy
		her_replay = strategy.get_her_transitions(standard_replay)

		## normalize standard transitions as well
		normalized_stnd_replay = []
		for transition in standard_replay:
			normalized_state = normalizer(transition[0][:-3], 5.0)
			normalized_goal = normalizer(transition[0][-3:], 5.0)
			normalized_next_state = normalizer(transition[3][:-3], 5.0)
			normalized_action = normalizer(transition[1], 5.0)

			normalized_stnd_replay.append([(np.concatenate((normalized_state, normalized_goal), axis=0)),
										   normalized_action, 
										   transition[2], 
										   (np.concatenate((normalized_next_state, normalized_goal), axis=0))])

		## store standard and HER replay transitions in Agent's memory
		agent.memory.store(normalized_stnd_replay)
		agent.memory.store(her_replay)


		print ("Epoch: {}, Episode: {}, Reward: {}".format(epoch+1, episode+1, episode_reward))
		episode_rewards.append(episode_reward)

print(episode_rewards)
plt.plot(episode_rewards)
plt.xticks(np.arange(0, len(episode_rewards), NUM_OF_EPISODES))
plt.show()
plt.savefig('200_800_50.png')



