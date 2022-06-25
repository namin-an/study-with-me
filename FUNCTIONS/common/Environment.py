# Reference: https://github.com/namin-an/pytorch-DRL/blob/master/common/Model.py

import os
import torch
import torch.nn as nn



class Env:
    def __init__(self, device, net, optimizer, last_trans, criterion, checkpoint_file_agents, mask_path, WINDOW_SIZE, SUBJECT_NUM, FOLD_NUM):
        self.device = device
        self.net = net
        self.optimizer = optimizer
        self.last_trans = last_trans
        self.criterion = criterion
        self.checkpoint_file_agents = checkpoint_file_agents
        self.mask_path = mask_path
        self.WINDOW_SIZE = WINDOW_SIZE
        self.SUBJECT_NUM = SUBJECT_NUM
        self.FOLD_NUM = FOLD_NUM
        
    def reset(self, t, features, mask):
        """
        resets states for the agent to make the useful action
        """
        states = features[:, t:t+self.WINDOW_SIZE] * mask[:, t:t+self.WINDOW_SIZE]
        return states
    
    def step(self, episode, t, actions, states, inputs, targets, mask, max_reward):
        
        mask[:, t] = actions
        features = self.net.extraction(inputs) 
        assert features.shape == mask.shape
        new_features = features * mask 
        
        if (t >= 0 and t <= (features.shape[-1]-self.WINDOW_SIZE)): 
            # Loss_FM -> ERROR
            outputs = self.net.classification(new_features)    
            loss_AM = self.criterion(outputs, targets.squeeze())  
            outputs = self.last_trans(outputs)
            probs, preds = torch.max(outputs.detach(), dim=1)
            acc = (preds == targets.squeeze()).sum() / float(preds.shape[0])
            
            reward = acc
            rewards = torch.tile(reward, (features.shape[0], 1))
            
            self.optimizer.zero_grad()
            loss_AM.backward()
            self.optimizer.step()  
            
            if (t > 0 and max_reward.item() <= reward.item()):              
                torch.save(self.net.state_dict(), self.checkpoint_file_agents)
                
            if (t == features.shape[-1]-self.WINDOW_SIZE):
                next_states, rewards, done, loss_AM = None, None, True, None
            else:
                done = False            
                next_states = self.reset(t+1, features, mask)  
                next_states = next_states - states
            
        return next_states, rewards, done, loss_AM, mask
