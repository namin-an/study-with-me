import torch
import torch.nn as nn
import torch.nn.functional as F

class ConstrainedConv2d(nn.Conv2d):
    def __init__(self, *args, max_norm, **kwargs):
        super(ConstrainedConv2d, self).__init__(*args, **kwargs) # initialization for nn.Conv2d
        self.max_norm = max_norm


    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(ConstrainedConv2d, self).forward(x)