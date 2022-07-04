# Reference:
# - https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py#L112


import os
import time
import random
import gc
import numpy as np
from itertools import count
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from utils.average_meter import AverageMeter
from utils.early_stop import EarlyStopping
    
from MARL.common.Model import DQN
from MARL.common.Memory import ReplayMemory
from MARL.common.Agent import Agent
from MARL.common.Environment import Env
    
    
class Experiment():
    def __init__(self, train_dataloader, valid_dataloader, net, learning_rate, num_epochs, num_epochs_pre, device0, device1, device2, checkpoint_file, log_path, model_type, class_num):
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.net = net
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_epochs_pre = num_epochs_pre
        self.device0, self.device1, self.device2 = device0, device1, device2
        self.checkpoint_file = checkpoint_file
        self.log_path = log_path
        self.model_type = model_type
        self.class_num = class_num
        
        self.writer = SummaryWriter(self.log_path) 
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.num_epochs)
        self.last_trans = nn.Softmax(dim=1) # along each data
        self.criterion = nn.CrossEntropyLoss()
        
        self.early_stopping = EarlyStopping(patience=self.num_epochs_pre, verbose=False, delta=0, checkpoint_file=self.checkpoint_file)
    
    # normal training procedure
    def train(self):
        device = self.device0
        
        if not os.path.isfile(self.checkpoint_file): 
            net = self.net.to(device)
            final_valid_accs = []
            
            for epoch in range(self.num_epochs_pre):
                losses, accs, accs2, stds, kappas = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

                # train
                self.net.train()
                for batch, data in enumerate(self.train_dataloader):
                    self.optimizer.zero_grad() # initialize model parameters

                    inputs = data['features'].to(device, dtype=torch.float)
                    targets = data['labels'].to(device, dtype=torch.long)

                    outputs = self.net(inputs)  # (batch_size, class_num)
                    loss = self.criterion(outputs, targets.squeeze())
                    outputs = self.last_trans(outputs)
                    probs, preds = torch.max(outputs.detach(), dim=-1)
                    probs2, preds2 = torch.topk(outputs.detach(), k=self.class_num//2, dim=-1)

                    if self.model_type == 'MSNN': # kernel regularizer (L1 + L2)
                        for name, param in self.net.named_parameters():
                            loss = loss + torch.norm(param, p=1)*0.001 + torch.norm(param, p=2)*0.01 
                    loss.backward() # back-propagation
                    self.optimizer.step() # update model parameters                
                    losses.update(loss.item(), inputs.size(0))

                    acc = (preds == targets.squeeze()).sum() / float(preds.shape[0])
                    accs.update(acc, inputs.size(0))

                    rep_targets = targets.repeat(1, self.class_num//2)
                    temp_bools = torch.eq(preds2, rep_targets)
                    temp_bins = temp_bools.int()
                    temp_bins = torch.sum(temp_bins, dim=1) # sum of "True's" along each data 
                    accs2.update(sum(temp_bins)/len(temp_bins), inputs.size(0))

                    kappa = (acc - 1/self.class_num) / (1 - 1/self.class_num)
                    kappas.update(kappa, inputs.size(0))

                self.writer.add_scalar('loss/train', losses.avg, epoch)
                self.writer.add_scalar('accuracy/train', accs.avg, epoch)
                self.writer.add_scalar(f'top{self.class_num//2}_accuracy/train', accs2.avg, epoch)
                self.writer.add_scalar('kappa/train', kappas.avg, epoch)
                self.scheduler.step()

                valid_losses, valid_accs, valid_accs2, valid_stds, valid_kappas = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

                # validation
                self.net.eval()
                for batch, data in enumerate(self.valid_dataloader):                
                    inputs = data['features'].to(device, dtype=torch.float)
                    targets = data['labels'].to(device, dtype=torch.long)

                    outputs = self.net(inputs) 
                    loss = self.criterion(outputs, targets.squeeze())   
                    outputs = self.last_trans(outputs)
                    probs, preds = torch.max(outputs.detach(), dim=1) # max, max_indices
                    probs2, preds2 = torch.topk(outputs.detach(), k=self.class_num//2, dim=1)

                    valid_losses.update(loss.item(), inputs.size(0))

                    acc = (preds == targets.squeeze()).sum() / float(preds.shape[0])
                    valid_accs.update(acc, inputs.size(0))
                    final_valid_accs.append(acc)

                    rep_targets = targets.repeat(1, self.class_num//2)
                    temp_bools = torch.eq(preds2, rep_targets)
                    temp_bins = temp_bools.int()
                    temp_bins = torch.sum(temp_bins, dim=1) # sum of "True's" along each data 
                    valid_accs2.update(sum(temp_bins)/len(temp_bins), inputs.size(0))

                    kappa = (acc - 1/self.class_num) / (1 - 1/self.class_num)
                    valid_kappas.update(kappa, inputs.size(0))

                self.writer.add_scalar('loss/valid', valid_losses.avg, epoch)
                self.writer.add_scalar('accuracy/valid', valid_accs.avg, epoch)
                self.writer.add_scalar(f'top{self.class_num//2}_accuracy/valid', valid_accs2.avg, epoch)
                self.writer.add_scalar('kappa/valid', valid_kappas.avg, epoch)

                if epoch % 30 == 0:
                    print(f'Epoch: {epoch:}/{self.num_epochs_pre}] Loss: {losses.avg:.3f} (valid loss: {valid_losses.avg:.3f}) Acc.: {accs.avg:.3f} (valid acc.: {valid_accs.avg:.3f})')

                self.early_stopping(valid_losses.avg, self.net)
                if self.early_stopping.early_stop:
                    print("Early stopped.")
                    break

            self.writer.flush()
            self.writer.close()
        else:
            final_valid_accs = [0]
        return max(final_valid_accs), self.net
    
    
    # agent making the "mask" and fine-tuning the original model
    def train_agents(self, net, WINDOW_SIZE, STRIDE, NUM_ACTIONS, EPSILON, SUBJECT_NUM, FOLD_NUM, mask_path):  
        device = self.device0
        checkpoint_file_agents = self.checkpoint_file + '_agents' 
        
        if not os.path.isfile(checkpoint_file_agents):
            net.load_state_dict(torch.load(self.checkpoint_file))
            net = net.to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
            
            # Initialize replay memory "Experience" to capacity 10,000
            Experience = namedtuple("Experience", ("state", "action", "reward", "next_state"))
            memory = ReplayMemory(100000, Experience)
            
            # Initialize action-value and target action-value network   
            policy_net = DQN(WINDOW_SIZE, NUM_ACTIONS).to(device)
            target_net = DQN(WINDOW_SIZE, NUM_ACTIONS).to(device)
            
            optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=0.0003)
            criterion_policy = nn.SmoothL1Loss()
            target_net.load_state_dict(policy_net.state_dict())
            policy_net.train() 
            target_net.train()   
            
            early_stopping_agents = EarlyStopping(patience=self.num_epochs - self.num_epochs_pre, verbose=False, delta=0, checkpoint_file=checkpoint_file_agents)
            scheduler_agents = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.num_epochs - self.num_epochs_pre)
            
            for episode in range(self.num_epochs - self.num_epochs_pre): 
                Losses_FM, Losses_AM, ValidLosses = AverageMeter(), AverageMeter(), AverageMeter()    
                Rewards, Qvalues, Ratios, QLosses = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
                done = False
                
                # train
                self.net.train()
                for i, data in enumerate(self.train_dataloader):  
                    
                # data = next(iterated_dataloader) # fresh whole dataset everytime the episode starts
                    inputs = data['features'].to(device, dtype=torch.float)
                    self.writer.add_graph(self.net, inputs)
                    features = self.net.extraction(inputs)
                    targets = data['labels'].to(device, dtype=torch.long)

                    env = Env(device, inputs, targets, net, optimizer, self.last_trans, checkpoint_file_agents, mask_path, WINDOW_SIZE, SUBJECT_NUM, FOLD_NUM)
                    EPSILON * 0.99 if EPSILON >= 0.05 else EPSILON     
                    agent = Agent(device, n_actions=NUM_ACTIONS, epsilon=EPSILON) # masking (0), remaining (1)

                    # Initialize preprocessed states (batch, window_size)
                    mask, rewards = torch.ones(features.shape).to(device), torch.tensor([-1.])
                    states = env.reset(0, features)

                    # Until our agent learns how to select "informative" features do
                    for t in range(0, features.shape[-1], STRIDE):      
                        # Select actions (batch, 1)
                        actions = agent.act(states, policy_net, states.shape[0], device)

                        # Execute action and observe reward and the next state
                        # and update the original network with selected features   
                        next_states, rewards, done, loss_FM, loss_AM, mask = env.step(episode, t, actions, states, inputs, features, mask)   
                        if done:
                            break
                        Losses_FM.update(loss_FM)
                        Losses_AM.update(loss_AM)
                        Rewards.update(torch.mean(rewards))                    

                        # Store experiences in the replay memory
                        memory.push(states.shape[0], states, actions, rewards, next_states) # DQN update (affects the agent.act(state, policy_net) code)
                        # Optimize policy network using the memory buffer
                        if len(memory) >= 100:
                            # Sample random minibatch of experiences from the replay memory
                            experiences = memory.sample(32) # ex. [Experience(states=4, actions=5), Experience(states=4, actions=5)] # sampling one is enough, since one memory contains the whole batch.
                            batch = Experience(*zip(*experiences)) # ex. Experience(states=(4, 4), actions=(5, 5))         
                            # Convert tuples to tensors
                            state_batch = torch.stack(list(batch.state), dim=0).to(device) 
                            action_batch = torch.stack(list(batch.action), dim=0).to(device) 
                            reward_batch = torch.stack(list(batch.reward), dim=0).to(device)  
                            next_state_batch = torch.stack(list(batch.next_state), dim=0).to(device)
                            Q_values = torch.tensor([policy_net(state_batch)[i][action_batch[i]].item() for i in range(action_batch.shape[0])]).to(device)
                            Q_values = Q_values.squeeze()
                            

                            if t == features.shape[-1] - states.shape[-1]:
                                next_Q_values = reward_batch.to(device)
                            else:
                                # Select the optimal value Q at the next time-step
                                # by choosing the maximum Q-values among all possible actions
                                max_Q_values_next = torch.tensor([target_net(next_state_batch)[i].max(dim=-1)[0] for i in range(action_batch.shape[0])]).to(device) # max_a Q(s, a) for DQN 
                                reward_batch, max_Q_values_next = reward_batch.squeeze(), max_Q_values_next.squeeze()
                                next_Q_values = reward_batch + (max_Q_values_next * 0.99) # r_t + gamma * Q_(t+1)
                            next_Q_values = next_Q_values.squeeze()

                            max_Q_values_current = torch.tensor([policy_net(state_batch)[i].max(dim=-1)[0] for i in range(action_batch.shape[0])]).to(device)  # maximum predicted action-value
                            Qvalues.update(torch.mean(max_Q_values_current)) 
                            loss_q = criterion_policy(Q_values, next_Q_values)
                            loss_q = loss_q.clone().detach()
                            loss_q.requires_grad = True
                            optimizer_policy.zero_grad()
                            loss_q.backward()
                            QLosses.update(loss_q, inputs.size(0))
                            optimizer_policy.step()     
                            
                            ratio = torch.count_nonzero(mask).item() / (mask.shape[0]*mask.shape[1]) # calculate ratio only for the final mask
                            Ratios.update(ratio, inputs.size(0))
                        
                    # Update states
                    states = next_states   
                    
                    scheduler_agents.step()
                    
                    print(f'EPISODE: {episode}/{(self.num_epochs - self.num_epochs_pre)}, BATCH: {i}/{(len(self.train_dataloader))}, Loss FM: {Losses_FM.avg.item()}, Loss AM: {Losses_AM.avg.item()}, Reward: {Rewards.avg.item()}, Q-value: {Qvalues.avg.item()}, Q Loss: {QLosses.avg.item()}') # total reward and average maximum Q-value
                
                self.scheduler.step()
                                    
                # validation
                self.net.eval()
                for j, data in enumerate(self.valid_dataloader):                
                    inputs = data['features'].to(device, dtype=torch.float)
                    targets = data['labels'].to(device, dtype=torch.long)

                    outputs = self.net(inputs) 
                    valid_loss = self.criterion(outputs, targets.squeeze())   
                    ValidLosses.update(valid_loss, inputs.size(0))
                    
                    print(f'EPISODE: {episode}/{(self.num_epochs - self.num_epochs_pre)},BATCH: {j}/{len(self.valid_dataloader)}, Valid_loss:{ValidLosses.avg.item()}')
                
                early_stopping_agents(ValidLosses.avg.item(), self.net)
                if early_stopping_agents.early_stop:
                    print("Early stopped.")
                    break
                    
                self.writer.add_scalar(f'loss_agents/train/FM', Losses_FM.avg, episode)   
                self.writer.add_scalar(f'loss_agents/train/AM', Losses_AM.avg, episode)   
                self.writer.add_scalar(f'loss_agents/valid', ValidLosses.avg, episode)  
                # self.writer.add_image(f'mask', mask.unsqueeze(axis=0), epsiode) 
                self.writer.add_scalar(f'rewards', Rewards.sum, episode)
                self.writer.add_scalar(f'qvalues', Qvalues.avg, episode) # average maximum predicted action value
                self.writer.add_scalar(f'ratios', Ratios.avg, episode)
                self.writer.add_scalar(f'loss_agents/DQN', QLosses.avg, episode)
    
                if episode % 2 == 0:
                    target_net.load_state_dict(policy_net.state_dict())                                 

            self.writer.flush()
            self.writer.close()
            