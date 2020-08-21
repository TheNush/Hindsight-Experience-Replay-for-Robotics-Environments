import random
import numpy as np
from utils import get_reward, normalizer

class Final():

	def __init__(self, env):
		'''
		This is the FINAL strategy : 

		HER goal state is now the final state of the episode. 
			=> g_episode = S_terminal 

		'''
		self.env = env

	def get_her_transitions(self, stnd_replay):
		new_goal = stnd_replay[-1][3][:3]	## get the terminal state
	## THIS STRATEGY IS NOT YET IMPLEMENTED ##

class Future ():

	def __init__ (self, env, k=4):
		'''
		This is the FUTURE strategy : 

		For each transition pick any k other states that were
		achieved in the episode after reaching the state in
		the given transition
		Using these k states, create k more transitions to be
		used for HER

		'''

		self.env = env
		self.k = k

	def get_her_transitions(self, stnd_replay):
		'''
		Params : 
		@ stnd_replay : base transition that actually occured

		'''
		new_transitions = []
		for i in range(len(stnd_replay)):
			try:
				samples = random.sample(stnd_replay[i+1:], self.k)
			except:
				'''
					return everything remaining if sample
					population is less than k

				''' 
				samples = stnd_replay[i+1:]

			for sample in samples:
				new_goal = np.asarray(sample[3][:3])

				for transition in stnd_replay[i:]:
					'''
					If the current transition being looked at 
					goes to the new goal state, then break out 
					of the loop after setting reward to 0.0
					Else create a new transition with new goal
					and add it to the trajectory
					
					'''
					new_state = transition[0][:-3]
					new_next_state = transition[3][:-3]
					
					if (np.all(new_next_state[:3] == new_goal)):
						new_reward = 0.0
						break
					else:
						new_reward = get_reward(self.env, new_next_state[:3], new_goal)

					## normalize values before concatenating
					new_goal = normalizer(new_goal)
					new_state = normalizer(new_state, 5.0)
					new_next_state = normalizer(new_next_state, 5.0)
					action = normalizer(transition[1], 5.0)

					new_state = np.concatenate((new_state, new_goal), axis=0)
					new_next_state = np.concatenate((new_next_state, new_goal), axis=0)

					new_transition = [new_state, action, new_reward, new_next_state]

					new_transitions.append(new_transition)

		return (new_transitions)










