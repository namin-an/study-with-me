# Reference:
# - https://github.com/pytorch/examples/blob/main/reinforcement_learning/actor_critic.py#L112
#

import os
import time
import random
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
        
        self.early_stopping = EarlyStopping(patience=self.num_epochs_pre//10, verbose=False, delta=0, checkpoint_file=self.checkpoint_file)
    
    # normal training procedure
    def train(self):
        if not os.path.isfile(self.checkpoint_file): 

            final_valid_accs = []
            for epoch in range(self.num_epochs_pre):
                losses, accs, accs2, stds, kappas = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

                # train
                self.net.train()
                for batch, data in enumerate(self.train_dataloader):
                    optimizer.zero_grad() # initialize model parameters

                    inputs = data['features'].to(self.device0, dtype=torch.float)
                    targets = data['labels'].to(self.device0, dtype=torch.long)

                    outputs = self.net(inputs)  # (batch_size, class_num)
                    outputs = self.last_trans(outputs)
                    probs, preds = torch.max(outputs.detach(), dim=-1)
                    probs2, preds2 = torch.topk(outputs.detach(), k=self.class_num//2, dim=-1)

                    loss = self.criterion(outputs, targets.squeeze())
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
                scheduler.step()

                valid_losses, valid_accs, valid_accs2, valid_stds, valid_kappas = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

                # validation
                self.net.eval()
                for batch, data in enumerate(self.valid_dataloader):                
                    inputs = data['features'].to(self.device0, dtype=torch.float)
                    targets = data['labels'].to(self.device0, dtype=torch.long)

                    outputs = self.net(inputs) 
                    outputs = self.last_trans(outputs)
                    probs, preds = torch.max(outputs.detach(), dim=1) # max, max_indices
                    probs2, preds2 = torch.topk(outputs.detach(), k=self.class_num//2, dim=1)

                    loss = self.criterion(outputs, targets.squeeze())                
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

                if epoch % 30 == 0: # cumulative values
                    print(f'Epoch: {epoch:}/{self.num_epochs_pre}] Loss: {losses.avg:.3f} (valid loss: {valid_losses.avg:.3f}) Acc.: {accs.avg:.3f} (valid acc.: {valid_accs.avg:.3f})')

                self.early_stopping(valid_losses.avg, self.net)
                if self.early_stopping.early_stop:
                    print("Early stopped.")
                    break

            self.writer.flush()
            self.writer.close()
        else:
            final_valid_accs = [0]
        return self.net, max(final_valid_accs)
    
    
    # fine-tuning with our agent module
    def train_agents(self, net, BATCH_SIZE, WINDOW_SIZE, NUM_ACTIONS, EPSILON, SUBJECT_NUM, FOLD_NUM):  
        
        checkpoint_file_agents = self.checkpoint_file + '_agents'    
        if not os.path.isfile(checkpoint_file_agents):
            net.load_state_dict(torch.load(self.checkpoint_file))
            net = net.to(self.device1) 
            
            # Initialize replay memory "Experience" to capacity 10,000
            Experience = namedtuple("Experience", ("state", "action", "reward", "next_state"))
            memory = ReplayMemory(10000, Experience)
            
            # Initialize action-value and target action-value network   
            policy_net = DQN(WINDOW_SIZE, NUM_ACTIONS).to(self.device1)
            target_net = DQN(WINDOW_SIZE, NUM_ACTIONS).to(self.device1)
            print(policy_net)
            
            optimizer_policy = torch.optim.RMSprop(policy_net.parameters(), lr=0.001, momentum=0.95)
            criterion_policy = nn.SmoothL1Loss()
            target_net.load_state_dict(policy_net.state_dict())
            policy_net.train() 
            target_net.train()            
            
            for episode in range(self.num_epochs - self.num_epochs_pre): 
                if episode % (len(self.train_dataloader)) == 0:
                    iterated_dataloader = iter(self.train_dataloader)
                Losses, Rewards, Qvalues = AverageMeter(), AverageMeter(), AverageMeter()                
                done = False
                
                env = Env(self.device1, net, self.criterion, checkpoint_file_agents, WINDOW_SIZE, SUBJECT_NUM, FOLD_NUM)
                EPSILON * 0.995 if EPSILON > 0.1 else EPSILON     
                agent = Agent(self.device1, n_actions=NUM_ACTIONS, epsilon=EPSILON) # masking (0), remaining (1)
                
                data = next(iterated_dataloader) # fresh whole dataset everytime the episode starts
                inputs = data['features'].to(self.device1, dtype=torch.float)
                features = net.extraction(inputs)
                targets = data['labels'].to(self.device1, dtype=torch.long)

                # Initialize preprocessed states (batch, window_size)
                mask, rewards = torch.ones(features.shape).to(self.device1), torch.tensor([-1.])
                states = env.reset(0, features, mask)

                # Until our agent learns how to select "informative" features do
                for t in count():
                    # Select actions (batch, 1)
                    actions = agent.act(states, policy_net, BATCH_SIZE, self.device1) 

                    # Execute action and observe reward and the next state
                    # and update the original network with selected features   
                    next_states, rewards, done, loss_AM, mask = env.step(episode, t, actions, states, features, targets, BATCH_SIZE, mask, Rewards.max_val)   
                    self.writer.add_image(f'mask/train/episode{episode}', mask.unsqueeze(axis=0), t)                
                    if done:
                        break
                    Losses.update(loss_AM) 
                    Rewards.update(torch.mean(rewards))                    

                    # Store experiences in the replay memory
                    memory.push(states, actions, rewards, next_states) # DQN update (affects the agent.act(state, policy_net) code)
                    # Optimize policy network using the memory buffer
                    if len(memory) >= 1:
                        # Sample random minibatch of experiences from the replay memory
                        experiences = memory.sample(1) # ex. [Experience(states=4, actions=5), Experience(states=4, actions=5)] # sampling one is enough, since one memory contains the whole batch.
                        batch = Experience(*zip(*experiences)) # ex. Experience(states=(4, 4), actions=(5, 5))         

                        # Convert tuples to tensors
                        state_batch = torch.cat(batch.state).to(self.device1) 
                        action_batch = torch.cat(batch.action).to(self.device1) 
                        reward_batch = torch.cat(batch.reward).to(self.device1)  
                        next_state_batch = torch.cat(batch.next_state).to(self.device1)

                        Q_values = torch.tensor([policy_net(state_batch)[i][action_batch[i]].item() for i in range(action_batch.shape[0])]).to(self.device1) 
                        Q_values = Q_values.squeeze()

                        if t == features.shape[-1] - states.shape[-1]:
                            next_Q_values = reward_batch.to(self.device1)
                        else:
                            # Select the optimal value Q at the next time-step
                            # by choosing the maximum Q-values among all possible actions
                            max_Q_values = torch.tensor([target_net(next_state_batch)[i].max(dim=-1)[1] for i in range(action_batch.shape[0])]).to(self.device1) # max Q(s, a) for DQN 
                            reward_batch, max_Q_values = reward_batch.squeeze(), max_Q_values.squeeze()
                            next_Q_values = reward_batch + (max_Q_values * 0.99) # r_t + gamma * Q_(t+1)
                        next_Q_values = next_Q_values.squeeze()

                        Qvalues.update(torch.mean(Q_values)) 
                        loss_q = criterion_policy(Q_values, next_Q_values)
                        loss_q = loss_q.clone().detach()
                        loss_q.requires_grad = True
                        optimizer_policy.zero_grad()
                        loss_q.backward()
                        optimizer_policy.step()

                    # Update states
                    states = next_states
                    self.writer.add_scalar(f'loss_AM/train/episode{episode}', loss_AM, t)                
                    self.writer.add_scalar(f'reward_AM/train/episode{episode}', torch.mean(rewards), t)
                    self.writer.add_scalar(f'qvalues_agents/train/episode{episode}', torch.mean(Q_values), t)
                
                self.writer.add_scalar(f'loss_AM/train', Losses.avg, episode)                
                self.writer.add_scalar(f'reward_AM/train', Rewards.avg, episode)
                self.writer.add_scalar(f'qvalues_agents/train', Qvalues.avg, episode)
                    
                # Update the parameters in the target network 
                if episode % 2 == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                                  
                print(f'Episode: {episode:}/{self.num_epochs - self.num_epochs_pre}, Rewards: {Rewards.avg:.4f}, Q-values: {Qvalues.avg:.4f}')                 
                
                del Losses, Rewards, Qvalues, env, agent, data, inputs, features, targets, mask, rewards, states, experiences, batch, state_batch, action_batch, reward_batch, next_state_batch, Q_values, next_Q_values, max_Q_values, loss_q

                torch.cuda.empty_cache()

            self.writer.flush()
            self.writer.close()
        
        
    def finetune(self, mask_path):    
        self.net.to(self.device0)
        self.net.load_state_dict(torch.load(self.checkpoint_file)) # fine-tune
        
        mask_files = []
        for file in os.listdir(mask_path):
            name, ext = file.split('.')
            split_name = name.split('_')
            if ext == 'pt':
                mask_files.append(os.path.join(mask_path, file))
        
        # we will update finetune the model based on the mask created by our agent
        checkpoint_file_agents = self.checkpoint_file + '_agents'
        if not os.path.isfile(checkpoint_file_agents):
            for mask_file in mask_files:
                mask = torch.load(mask_file)
                mask = mask.to(self.device0)
                
                final_valid_accs = []
                self.net.train()

                iterated_dataloader = iter(self.train_dataloader)
                data = next(iterated_dataloader) # whole data (230 trials)
                inputs = data['features'].to(self.device0, dtype=torch.float)
                targets = data['labels'].to(self.device0, dtype=torch.long)

                self.optimizer.zero_grad() # initialize model parameters

                features = self.net.extraction(inputs)
                assert features.shape == mask.shape
                masked_features = features * mask            
                outputs = self.net.classification(masked_features)  
                outputs = self.last_trans(outputs)

                loss = self.criterion(outputs, targets.squeeze())
                loss.backward() 
                self.optimizer.step()   
            
            torch.save(self.net.state_dict(), checkpoint_file_agents)