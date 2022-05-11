#!/usr/bin/env python
# coding: utf-8

# References:
# - https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
# - https://somjang.tistory.com/entry/Google-Colab%EC%97%90%EC%84%9C-OpenAI-gym-render-%EC%82%AC%EC%9A%A9%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95

# # Importing packages
# from google.colab import drive
# drive.mount('/content/gdrive')
import os
# os.getcwd()
# get_ipython().system(' pip install nes-py')
# get_ipython().system('pip install gym-super-mario-bros==7.3.0')
import time, datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros


# # Initializing and preprocessing environment
class SkipFrame(gym.Wrapper):
  """
  Skipping n-intermediate frames w/ losing much information.
  """
  def __init__(self, env, skip):
    """Returning only every 'skip'-th frame"""
    super().__init__(env)
    self._skip = skip

  def step(self, action):
    """Summing up reward with the given action"""
    total_reward = 0.0
    done = False
    for i in range(self._skip):
      # Acumulate reward and repeat the same action
      obs, reward, done, info = self.env.step(action)
      total_reward += reward # stands for the "quality" of the action in a state
      if done:
        self.env.reset()
        break
    return obs, total_reward, done, info

class GrayScaleObservation(gym.ObservationWrapper):
    """
    Turning environment into gray-scaled space.
    """
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        """permute [H, W, C] array to [C, H, W] tensor"""
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        """return grayscaled observation space"""
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

class ResizeObservation(gym.ObservationWrapper):
    """
    Resizing environment to the given shape
    """
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else: # if shape is not an integer
            self.shape = tuple(shape)
        
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
    
    def observation(self, observation):
        """return resized and normalized space"""
        transforms = T.Compose([T.Resize(self.shape), T.Normalize(0, 255)])
        observation = transforms(observation).squeeze(0)
        return observation


# # Creating agent Mario for reinforcement learning
# Creating agent's DNN
class MarioNet(nn.Module):
    """
    Creating mini cnn structure for Mario
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        
        if h != 84:
            raise ValueError(f'Expecting input height: 84, got : {h}')
        if w != 84:
            raise ValueError(f'Expecting input height: 84, got : {w}')

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        # Freeze parameters for target DQN
        for p in self.target.parameters():
            p.requires_grad = False
        
    def forward(self, input, model):
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)


# Creating agent
class Mario:
    """
    Creating Mario that can act wisely by choosing the most optimal action (exploit) or takes a random action (explore).
    """
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # cache and recall
        self.memory = deque(maxlen=10000)

        # learn
        self.batch_size = 32
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.curr_step = 0
        self.burnin = 1e4 # minimum no. of experiences before training
        self.learn_every = 3 # no. of experiences between updates to Q_online
        self.sync_every = 1e4 # no. of experiences between Q_target & Q_online sync

        self.save_every = 5 #5e5 # no. of experiences between saving MarioNet
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available() # True
        
        # Mario's DNN to predict the most optimal action
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')
        if checkpoint:
            self.load(checkpoint)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
      
    def act(self, state):
        """perform a greedy action based on epsilon value given a state"""
        # EXPLORE
        if np.random.randn() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        # EXPLOIT
        else:        
            state = state.__array__()
            state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()
            
        # decrease exploration rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        
        # increment step
        self.curr_step += 1
        return action_idx  
        
    def cache(self, state, next_state, action, reward, done): # experiences
        """stores experiences to memory (replay buffer)"""
        state = state.__array__()
        next_state = next_state.__array__()
        
        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(state)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor(state)
        reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor(state)
        done = torch.BoolTensor([done]).cuda() if self.use_cuda else torch.BoolTensor(state)

        self.memory.append((state, next_state, action, reward, done,))
    
    def recall(self):
        """retrieves a batch of experiences from memory"""
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        """predict optimal Q-value for a given state"""
        current_Q = self.net(state, model='online')[np.arange(0, self.batch_size), action] # Q_online(s, a)
        return current_Q
      
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """aggregate current reward and the estimated Q-values for the next state"""
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1) # precalculating action value for the next state
        next_Q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
      
    def update_Q_online(self, td_estimate, td_target):
        """backpropagate online dqn"""
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        """copy online dqn to target dqn"""
        self.net.target.load_state_dict(self.net.online.state_dict()) 

    def save(self):
        """save checkpoint"""
        save_path = (self.save_dir / f'mario_net_{int(self.curr_step // self.save_every)}.chkpt')
        torch.save(dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate), save_path)
        print(f'MarioNet saved to {save_path} at step {self.curr_step}')

    def load(self, load_path):
        """load model with saved weights"""
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
    
    def learn(self):
        """update online action value (Q) function with a batch of experiences"""
        # ready to update target DQN
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        # ready to save
        if self.curr_step % self.save_every == 0:
            self.save()
        
        # not ready to train
        if self.curr_step < self.burnin:
            return None, None
        
        # sample from memory bank
        state, next_state, action, reward, done = self.recall()

        # get Q-values of online dqn
        td_est = self.td_estimate(state, action)

        # get Q-values of target dqn
        td_tgt = self.td_target(reward, next_state, done)

        # backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


# # Logging to save training information
class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Moving averages, added for every call to record()
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Mean Q Value {mean_ep_q} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()





