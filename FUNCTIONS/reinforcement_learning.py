import math
import random
import re
from collections import deque

import numpy as np
print("numpy: ", np.__version__)

import cv2 as cv
print("cv: ", cv.__version__)

import torch
import torch.nn as nn
from torch import save
from torch.optim import Adam
print("torch: ", torch.__version__)

import atari_py as ap

from gym import make, ObservationWrapper, Wrapper
from gym.spaces import Box

from FUNCTIONS.packages.set_environ import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("hardware device: ", device)



class QLearnModel():
    def __init__(self):
        """
        Initializing Q-learning parameters (actions, states and rewards).
        """
        
        zeros = [0.0, 0.0]
        self.Qvals = [zeros, zeros, zeros, zeros, zeros] # if all the training goes well, values of the second column (down) will have higher q-values.
        self.end_states = [1, 0, 0, 0, 1]
        self.rewards = [-1, 0, 0, 0, 2]

        self.num_episodes = 100
        self.epsilon = 1
        self.gamma = 0.9

    def act_greedy(self, epsilon, s):
        """
        Taking random action (off-policy) to Q-values to find optimal solution for an agent.
        """

        rand_num = np.random.uniform()
        if rand_num < epsilon:
            return np.random.randint(0, 2)
        else:
            return np.argmax(self.Qvals[s]) # either 0 (up) or 1 (down). did not add random noise yet.
    
    def return_rs(self, a, s):
        """
        Automatically returning the next state and reward for the given action and the current state.
        """
        
        if a == 0: # move up
            s_next = s - 1
        else: # move down
            s_next = s + 1
        return self.rewards[s_next], s_next

    def qlearn(self):
        """
        Q-learning by taking greedy actions for each episode assuming we don't know transition probabilities of MDP.
        """
        
        traj_actions = list()
        for n in range(self.num_episodes):
            init_state = 2
            s = init_state
            while not self.end_states[s]: # until the initial state reaches the very bottom or the very top
                a = self.act_greedy(self.epsilon, s)
                traj_actions.append(a)
                r, s_next = self.return_rs(a, s)
                if self.end_states[s_next]: # if the state reached the end state (either at the very top or the very bottom)
                    self.Qvals[s][a] = r # it does not need to update Q-values anymore
                else:
                    self.Qvals[s][a] = r + self.gamma * max(self.Qvals[s_next]) # use bellman equation as an iterative update
                s = s_next
            self.epsilon = self.epsilon - 1 / self.num_episodes # as the number of episodes increase, Q-table is taken into the consideration rather than taking random action
        return self.Qvals, traj_actions
    


class DQLearnModel(nn.Module):
    def __init__(self, ip_sz, tot_num_acts):
        """
        Initializing Q-learning parameters (layers of learnable DNN).
        """
        
        super(DQLearnModel, self).__init__()
        
        self._ip_sz = ip_sz
        self._tot_num_acts = tot_num_acts
        
        self.cnv1 = nn.Conv2d(ip_sz[0], 32, kernel_size=8, stride=4)
        self.rl = nn.ReLU()
        self.cnv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.cnv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.feat_sv, 512) 
        self.fc2 = nn.Linear(512, self._tot_num_acts)
                
        self.num_eps = 5000 # total number of episodes to train agent
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
    def forward(self, x): # the shape of x: [1, 4, 84, 84]   
        x = x.to(self.device)
        
        op = self.cnv1(x)
        op = self.rl(op)
        op = self.cnv2(op)
        op = self.rl(op)
        op = self.cnv3(op)
        
        op = self.rl(op).view(x.size()[0], -1) # flattens the features
        
        op = self.fc1(op)
        op = self.rl(op)
        op = self.fc2(op)
        return op
    
    @property
    def feat_sv(self):
        """
        Automatically calculating the size of feature vector.
        """
        
        x = torch.zeros(1, *self._ip_sz)
        x = self.cnv1(x)
        x = self.rl(x)
        x = self.cnv2(x)
        x = self.rl(x)
        x = self.cnv3(x)
        x = self.rl(x)
        return x.view(1, -1).size(1) # s.view(-1, 1).size(0)
    
    def act_perf(self, stt, eps, device):
        """
        Random action or if trained enough, action with the largest Q-values.
        """
        
        if random.random() > eps:
            stt = torch.from_numpy(np.float32(stt)).unsqueeze(0).to(device)
            q_val = self.forward(stt)
            act = q_val.max(1)[1].item() # observes values along all columns and returns the argmax (action)
        else:
            act = random.randrange(self._tot_num_acts)
        return act
    
    
class ReplayBuffer:
    """
    Storing thousands of gaming frames and then sampling i.i.d. frames to train DNN models.
    """

    def __init__(self, max_cap):
        """
        Initializing the buffer of which length is fixed.
        """

        self._bfr = deque(maxlen=max_cap)

    def push(self, s, a, r, next_s, fin):
        """
        Adding basic components of RL in the buffer saved previously.
        """

        self._bfr.append((s, a, r, next_s, fin))

    def sample(self, batch_size):
        """
        Sampling batches from the buffer without replacement.
        """

        idxs = np.random.choice(len(self._bfr), batch_size, replace=False)
        batch = zip(*[self._bfr[i] for i in idxs]) # subset of selected random Q-values
        s, a, r, next_s, fin = batch
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32), np.array(next_s), np.array(fin, dtype=np.uint8))

    def __len__(self):
        """
        Returning the length of the buffer less than or equal to max_cap.
        """

        return len(self._bfr)

    
def ret_optimizer(model, alpha):
    """
    Returning Adam optimizer with the given learning rate.
    """

    return Adam(model.parameters(), lr=alpha)


def wrap_env(env_ip):
    """
    Using initialization and control functions of gaming environments to set up the final environment. 
    """

    env = make(env_ip)
    is_atari = check_atari_env(env_ip) # from hand-made package # 1
    env = CCtrl(env, is_atari) # from hand-made package # 2
    env = MaxNSkpEnv(env, is_atari) # from hand-made package # 3
    try:
        env_acts = env.unwrapped.get_action_meanings()
        if "FIRE" in env_acts:
            env = FrRstEnv(env) # from hand-made package # 4
    except AttributeError:
        pass
    env = FrmDwSmpl(env) # from hand-made package # 5
    env = Img2Trch(env) # from hand-made package # 6
    env = FrmBfr(env, 4) # from hand-made package # 7
    env = NormFlts(env) # from hand-made package # 8
    return env


def train_model(env, main_dql_model, tgt_dqn_model, opt, rep_buffer, device):
    """
    Training two backbone models to calculate Q-values for the present and the next (state, action) pairs. 
    """

    log = TrMetadata() # from hand-made package # 9
    # print(dir(log))
    for episode in range(50000): # num_epochs
        run_episode(env, main_dql_model, tgt_dqn_model, opt, rep_buffer, device, log, episode) # from hand-made package # 10

        
def init_models(env, device):
    """
    Initializing two model instances (but with same architecture) to stably train the models. 
    """

    main_DQLearnModel = DQLearnModel(env.observation_space.shape, env.action_space.n).to(device) # updates weights when trainig to generate target action-values
    tgt_DQLearnModel = DQLearnModel(env.observation_space.shape, env.action_space.n).to(device) # separate network for generating target values which can stabilize Q-learning process
    return main_DQLearnModel, tgt_DQLearnModel

