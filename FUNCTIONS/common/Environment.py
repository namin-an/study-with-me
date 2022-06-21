# Reference: https://github.com/namin-an/pytorch-DRL/blob/master/common/Model.py

import torch
import torch.nn as nn

class Env:
    def __init__(self, device, net, criterion, checkpoint_file_agents, WINDOW_SIZE):
        self.device = device
        self.net = net
        self.criterion = criterion
        self.checkpoint_file_agents = checkpoint_file_agents
        self.WINDOW_SIZE = WINDOW_SIZE
        
        
    def reset(self, t, features, mask):
        """
        resets states for the agent to make the useful action
        """
        states = features[:, t:t+self.WINDOW_SIZE] * mask
        return states
    
    def step(self, episode, t, actions, states, features, targets, BATCH_SIZE, mask, max_reward):
        
        # original loss on the pretrained model
        loss_FM = self.criterion(self.net.classification(features), targets.squeeze())
        
        mask[:, t] = actions
        new_states = torch.mul(states, mask)
        new_features = new_states
        
        assert features.shape == new_features.shape
        if (t >= 0 and t <= (features.shape[-1]-self.WINDOW_SIZE)): 
            loss_AM = self.criterion(self.net.classification(new_features), targets.squeeze())
            
            reward = loss_FM - loss_AM
            rewards = torch.tile(reward, (BATCH_SIZE, 1))
            if max_reward.item() < reward.item():
                torch.save(mask, f'/opt/pytorch/demo/masks/final_mask_{episode}.pt')

            if (t == features.shape[-1]-self.WINDOW_SIZE) or bool(reward.item() > 1.0 and loss_AM.item() < 1.0):
                next_states, rewards, done, loss_AM = None, None, True, None
            else:
                done = False
            
                next_states = self.reset(t+1, features, mask)  
                next_states = next_states - states
        else:
            next_states, rewards, done, loss_AM = None, None, True, None
            
        return next_states, rewards, done, loss_AM, mask
