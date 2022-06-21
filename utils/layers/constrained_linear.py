import torch
import torch.nn as nn


class ConstrainedLinear(nn.LazyLinear):
    def __init__(self, *args, max_norm, **kwargs): 
        super(ConstrainedLinear, self).__init__(*args, **kwargs) # initialization for nn.LazyLinear
        self.max_norm = max_norm # 0.5 for ShallowConvNet and DeepConvNet, 0.25 for EEGNets
        
    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm = self.max_norm
        )
        return self(x)
    
 