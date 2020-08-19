from models import *
from utils import *
from torch import nn
from torch import optim
import torch
from torch.autograd import Variable

class DDPGAgent ():
	def __init__ (self, env, hidden_size=64, actor_learning_rate=1e-3, critic_learning_rate=1e-3, gamma=0.98, tau=1e-2, max_memory=1000000):
		'''
		initialize all basic properties

		Params:

			env 					: environment variable
			hidden_size 			: hidden layer size for the actor and critic networks
			actor_learning_rate 	: actor's learning rate
			critic_learning_rate 	: critic's learning rate 			[NOTE: this is value was chosen to be higher than the actor's value... Why??]
			max_memory 				: maximum memory of the replay buffer

		'''
		# environment variables
		self.goal_shape = 3
		self.num_states = env.observation_space["observation"].shape[0] + self.goal_shape
		# self.num_states = env.observation_space.shape[0]
		self.num_actions = env.action_space.shape[0]
		self.gamma = gamma
		self.tau = tau

		# Networks
		self.critic = CriticNetwork(self.num_states + self.num_actions, hidden_size, 1)
		self.target_critic = CriticNetwork(self.num_states + self.num_actions, hidden_size, 1)
		self.actor = ActorNetwork(self.num_states, hidden_size, self.num_actions)
		self.target_actor = ActorNetwork(self.num_states, hidden_size, self.num_actions)

		# weights of original and target networks
		for params, target_params in zip(self.critic.parameters(), self.target_critic.parameters()):
			target_params.data.copy_(params.data)

		for params, target_params in zip(self.actor.parameters(), self.target_actor.parameters()):
			target_params.data.copy_(params.data)

		# replay buffer
		self.memory = Memory(max_memory)

		# optimizers & losses
		self.critic_loss = nn.MSELoss()
		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

	def get_action (self, state, goal):
		if (np.random.uniform(0,1) > 0.2):
			state = Variable (torch.from_numpy(state).float().unsqueeze(0))
			goal = Variable (torch.from_numpy(goal).float().unsqueeze(0))
			state_rep = torch.cat((state, goal), 1)
			action = self.actor.forward(state_rep)
			action = action.detach().numpy()[0]
			return action
		else:
			return None

	def update (self, batch_size):
		states, actions, rewards, next_states = self.memory.sample(batch_size)
		states = torch.FloatTensor(states)
		actions = torch.FloatTensor(actions)
		rewards = torch.FloatTensor(rewards).unsqueeze(1)
		next_states = torch.FloatTensor(next_states)

		# Evaluate critic loss
		Qvals = self.critic.forward(states, actions)

		next_actions = self.target_actor.forward(next_states)
		next_Qvals = self.target_critic.forward(next_states, next_actions.detach())
		target_Q = rewards + self.gamma * next_Qvals
		critic_loss = self.critic_loss(Qvals, target_Q)
		# print("Critic Loss: ", critic_loss)

		# Evaluate actor loss
		action_pred = self.actor.forward(states)
		actor_loss = -self.critic.forward(states, action_pred).mean()
		# actor_loss += (torch.atanh(action_pred/5.0))**2			## add square of the pre-activations to prevent tanh saturation and vanishing gradient
		# print("Actor Loss: ", actor_loss)

		# update original networks
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# update target networks
		for target_params, params in zip (self.target_critic.parameters(), self.critic.parameters()):
			target_params.data.copy_(params.data * self.tau + target_params.data * (1.0 - self.tau))

		for target_params, params in zip (self.target_actor.parameters(), self.actor.parameters()):
			target_params.data.copy_(params.data * self.tau + target_params.data * (1.0 - self.tau))

	def save_models(self):
		torch.save(self.actor.state_dict(), "C:/Users/dhanu/Desktop/Work/Projects/Fetch Reach/DDPG/saved_models/actor.pt")
		torch.save(self.target_actor.state_dict(), "C:/Users/dhanu/Desktop/Work/Projects/Fetch Reach/DDPG/saved_models/actor_target.pt")
		torch.save(self.critic.state_dict(), "C:/Users/dhanu/Desktop/Work/Projects/Fetch Reach/DDPG/saved_models/critic.pt")
		torch.save(self.target_critic.state_dict(), "C:/Users/dhanu/Desktop/Work/Projects/Fetch Reach/DDPG/saved_models/critic_target.pt")

	def load_models(self):
		self.actor.load_state_dict(torch.load("C:/Users/dhanu/Desktop/Work/Projects/Fetch Reach/DDPG/saved_models/actor.pt"))
		self.target_actor.load_state_dict(torch.load("C:/Users/dhanu/Desktop/Work/Projects/Fetch Reach/DDPG/saved_models/actor_target.pt"))
		self.critic.load_state_dict(torch.load("C:/Users/dhanu/Desktop/Work/Projects/Fetch Reach/DDPG/saved_models/critic.pt"))
		self.target_critic.load_state_dict(torch.load("C:/Users/dhanu/Desktop/Work/Projects/Fetch Reach/DDPG/saved_models/critic_target.pt"))
