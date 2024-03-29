import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import time
import random
import traceback
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.average_meter import AverageMeter
        

class Evaluate():
    def __init__(self, test_dataloader, net, device, checkpoint_files):
        self.test_dataloader = test_dataloader
        self.net = net
        self.device = device
        self.checkpoint_files = checkpoint_files

    def test(self):
        torch.cuda.empty_cache()
        self.net = self.net.to(self.device)
        test_accs = AverageMeter()
        
        for batch, data in enumerate(self.test_dataloader):
            
            inputs = data['features'].to(self.device, dtype=torch.float)
            targets = data['labels'].to(self.device, dtype=torch.long)
            preds_cands = []
            for checkpoint_file in self.checkpoint_files:
                try:
                    self.net.load_state_dict(torch.load(checkpoint_file, map_location=self.device))
                    outputs = self.net(inputs)  # (batch_size, class_num)
                    probs, preds = torch.max(outputs.detach(), dim=1)
                    preds_cands.append(preds.item())        
                except Exception as e:
                    print(checkpoint_file)
                    pass
            c = Counter(preds_cands) # {value: counted number} starting with the highest counted number value
            try:
                # ALWAYS CHOOSE THE FIRST ONE
                final_idxs = c.most_common()[0][0] 
                
                # CHOOSE THE RANDOM ONE
#                 ordered_cands = c.most_common()
#                 max_freq = ordered_cands[0][-1]
#                 final_idxs_cands = []
#                 for class_num, freq in ordered_cands:
#                     if freq == max_freq: 
#                         final_idxs_cands.append(class_num)
#                 final_idxs = random.choice(final_idxs_cands)

                test_accs.update((targets == final_idxs).float().mean(), inputs.size(0))
            except Exception as e:
                print(e)
                print(traceback.format_exc())
        return test_accs.avg            
         
