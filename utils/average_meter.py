import math
import torch

class AverageMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max_val = torch.tensor([-math.inf])
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count
#         if self.count == 1:
#             self.max_val = val
#         else:
#             if val > self.max_val:
#                 self.max_val = val
        
  