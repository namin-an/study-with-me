# Reference:
# - https://github.com/namin-an/pytorch-DRL/blob/master/common/Memory.py
# - https://velog.io/@wjleekr927/%EA%B0%95%ED%99%94-%ED%95%99%EC%8A%B5-DQN-%ED%8A%9C%ED%86%A0%EB%A6%AC%EC%96%BC-PyTorch

import random
from collections import namedtuple, deque



class ReplayMemory(object):
    def __init__(self, capacity, Experience):
        self.memory = deque([], maxlen=capacity) # FIFO & LIFO   
        self.Experience = Experience
        
    def push(self, *args):
        """
        Stores experiences in the memory buffer.
        """
        self.memory.append(self.Experience(*args))
    
    def sample(self, batch_size):
        """
        Returns experiences
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
   