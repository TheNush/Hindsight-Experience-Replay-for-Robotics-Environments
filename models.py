import torch
import torch.nn.functional as F

'''
This is this the file that contains all the model descriptions

'''

class ActorNetwork(torch.nn.Module):
	def __init__(self, input_shape, hidden_shape, output_shape):
		super(ActorNetwork, self).__init__()
		self.linear1 = torch.nn.Linear(input_shape, hidden_shape, bias=False)

		self.hidden1 = torch.nn.Linear(hidden_shape, hidden_shape, bias=False)
		self.hidden2 = torch.nn.Linear(hidden_shape, hidden_shape, bias=False)
		self.hidden3 = torch.nn.Linear(hidden_shape, hidden_shape, bias=False)

		self.linear2 = torch.nn.Linear(hidden_shape, output_shape, bias=False)

	def forward(self, state):
		'''
		Params: 
		@ state : state variable - used by the actor
				  to control update the policy offline.

		'''

		output = F.relu(self.linear1(state))

		output = F.relu(self.hidden1(output))
		output = F.relu(self.hidden2(output))
		output = F.relu(self.hidden3(output))

		output = torch.tanh(self.linear2(output))

		return output*5.0			## scale outputs to lie in the range of [-5, 5]

class CriticNetwork (torch.nn.Module):
	def __init__ (self, input_shape, hidden_shape, output_shape):
		super(CriticNetwork, self).__init__()
		self.linear1 = torch.nn.Linear(input_shape, hidden_shape, bias=False)

		self.hidden1 = torch.nn.Linear(hidden_shape, hidden_shape, bias=False)
		self.hidden2 = torch.nn.Linear(hidden_shape, hidden_shape, bias=False)
		self.hidden3 = torch.nn.Linear(hidden_shape, hidden_shape, bias=False)

		self.linear2 = torch.nn.Linear(hidden_shape, output_shape, bias=False)

	def forward(self, state, action):
		'''
		Params:
		@ state : state variable - same as the one above

		@ action : action variable - used to compute the 
				   policy gradient along with the state var

		'''
		output = torch.cat([state, action], 1) ## concatenate along the first dimension

		output = F.relu(self.linear1(output))

		output = F.relu(self.hidden1(output))
		output = F.relu(self.hidden2(output))
		output = F.relu(self.hidden3(output))

		output = (self.linear2(output))

		return output
