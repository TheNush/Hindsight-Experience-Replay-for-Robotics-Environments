import numpy as np
from collections import deque
import random

# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

'''
Memory class for using experience replay/replay buffers.

'''
class Memory:
	def __init__(self, max_size):
		self.max_size = max_size
		self.data = deque(maxlen=max_size)

	def sample (self, batch_size):
		'''
		Params:
		@ batch_size : number of data points to sample

		'''
		samples = random.sample(self.data, batch_size)
		states  = []
		actions = []
		rewards	= []
		next_states = []

		for experience in samples:
			states.append(experience[0])
			actions.append(experience[1])
			rewards.append(experience[2])
			next_states.append(experience[3])

		return states, actions, rewards, next_states

	def store (self, experiences):
		'''
		Params: 
		@ experiences : collection of (s,a,g,s') transitions

		'''
		for experience in experiences:
			self.data.append(experience)

	def __len__(self):
		return len(self.data)

def get_reward(env, achieved_goal, goal):
	return env.compute_reward(achieved_goal, goal, 'info')

def normalizer(data, clip_val=np.inf):
	mean = np.mean(data)
	std = np.std(data)

	return (np.clip((data-mean)/std, -clip_val, clip_val)) 


