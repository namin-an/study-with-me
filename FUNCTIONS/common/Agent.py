# Reference: https://jyoondev.tistory.com/m/156

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class Agent:
    def __init__(self, device, n_actions, epsilon):
        self.device = device
        self.n_actions = n_actions
        self.epsilon = epsilon

    def act(self, states, policy_net, BATCH_SIZE, device):
        """
        Explore or exploit
        """
        if np.random.randn() < self.epsilon:
            actions = np.random.choice(self.n_actions, size=BATCH_SIZE)
            actions = torch.from_numpy(actions).to(device)
        else:           
            with torch.no_grad():
                action_probs = policy_net(states)
                actions = action_probs.max(dim=-1)[1] # find the maximum index along all the action probabilities

        return actions
    