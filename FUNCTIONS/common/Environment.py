# Reference: https://github.com/namin-an/pytorch-DRL/blob/master/common/Model.py

import os
import torch
import torch.nn as nn


class Env:
    def __init__(self, device, inputs, targets, net, optimizer, last_trans, checkpoint_file_agents, mask_path, WINDOW_SIZE, SUBJECT_NUM, FOLD_NUM):
        self.device = device
        self.inputs = inputs
        self.targets = targets
        self.net = net
        self.optimizer = optimizer
        self.last_trans = last_trans
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.checkpoint_file_agents = checkpoint_file_agents
        self.mask_path = mask_path
        self.WINDOW_SIZE = WINDOW_SIZE
        self.SUBJECT_NUM = SUBJECT_NUM
        self.FOLD_NUM = FOLD_NUM
        
    def reset(self, t, features):
        """
        resets states for the agent to make the useful action
        """
        states = features[:, t:t+self.WINDOW_SIZE]
        return states
    
    def step(self, episode, t, actions, states, features, mask):

        mask[:, t:t+self.WINDOW_SIZE] = torch.tile(actions.unsqueeze(dim=-1), dims=(1, self.WINDOW_SIZE))
        
        temp1 = torch.ones(features.shape)
        temp1[:, t:t+self.WINDOW_SIZE] = torch.zeros(features[:, t:t+self.WINDOW_SIZE].shape)
        temp2 = torch.zeros(features.shape)
        temp2[:, t:t+self.WINDOW_SIZE] =  features[:, t:t+self.WINDOW_SIZE] * mask[:, t:t+self.WINDOW_SIZE]
        partial_mask = (temp1 + temp2).to(self.device)
        masked_features = features * partial_mask
                
        outputs = self.net.classification(masked_features)    
        del temp1, temp2, partial_mask, masked_features
        
        loss_AMs = self.criterion(outputs, self.targets.squeeze())  
        loss_AM = torch.mean(loss_AMs, dim=0)
        rewards = 1.5 - loss_AMs

        if (t >= features.shape[-1] - self.WINDOW_SIZE):
#             self.optimizer.zero_grad()   
#             loss_AM.backward()
#             self.optimizer.step() 
            next_states, rewards, done, loss_AM = None, None, True, None
        else:
            done = False            
            next_states = self.reset(t+1, features) # not affected by the current action  

        return next_states, rewards, done, loss_AM, mask
