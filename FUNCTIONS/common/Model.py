# Reference:
# - https://github.com/namin-an/pytorch-DRL/blob/master/common/Model.py\
# - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Returns probability of possible actions
    """
    def __init__(self, state_dim, n_actions):
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, n_actions)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

        
class Actor(nn.Module):
    """
    Returns probability of possible actions
    """
    def __init__(self, state_dim ,hidden_size, n_actions):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = self.softmax(self.fc3(out))
        return out

    
class Critic(nn.Module):
    """
    Returns Q-value
    """
    def __init__(self, state_dim, action_dim, hidden_size, output_size=1):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(action_dim, out_features=hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, state, action):
        out = F.relu(self.fc1(state))
        out = torch.cat([out, action], dim=-1)
        out = F.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))
        return out
    
    
class ActorCritic(nn.Module):
    """
    Returns probability of possible actions and Q-value
    """
    def __init__(self, state_dim, action_dim, hidden_size, critic_output_size=1):
        super(ActorCritic, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_fc3 = nn.Linear(hidden_size, actor_dim)
        self.critic_fc3 = nn.Linear(hidden_size, critic_output_size)
        
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        
    def __call__(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        action = self.softmax(self.actor_fc3(out))
        qvalue = self.sigmoid(self.critic_fc3(out))
        return action, qvalue

    
