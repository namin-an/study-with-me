# Reference:
# - https://github.com/fbcsptoolbox/fbcsp_code/blob/master/bin/MLEngine.py
# - https://github.com/ADD-Drone/drone-sjkim/blob/master/models/backbones/FBCSP.py
# - https://github.com/stupiddogger/FBCSP/blob/master/FBCSP.py

import time
from collections import Counter
import numpy as np
import pickle
import torch
import mne
from sklearn.svm import SVC
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
    
from utils.average_meter import AverageMeter
from FBCSP.FBCSP import FBCSP
    
class EvaluateFBCSP:
    def __init__(self, class_num, s_freq, l_freq, h_freq, width, step, m_filters:int=2):
        self.class_num = class_num
        self.s_freq = s_freq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.freq_range = [l_freq, h_freq]
        self.width = width
        self.step = step
        self.m_filters = m_filters    
        
    def test(self, test_dataloader, checkpoint_files):
        accs, times = AverageMeter(), AverageMeter()
        preds_cands = []
        
        # Iterate test data loader
        X_test, y_test = np.array([]), np.array([])
        for data in test_dataloader:
            inputs = data['features']
            targets = data['labels']
            
            if X_test.shape[0] == 0 and y_test.shape[0] == 0:
                X_test, y_test = inputs, targets
            else:
                X_test = np.concatenate([X_test, inputs], axis=0)
                y_test = np.concatenate([y_test, targets], axis=0)
        X_test, y_test = X_test.squeeze(), y_test.squeeze()
        
        start_time = time.time()
        for checkpoint_file in checkpoint_files:
            # Bandpass filtering (4~8, 8~12, ..., 36~40) (6-10, 10-14, ..., 34-38 added for over-lapping condition)
            filt_X_test = np.zeros((len(range(self.freq_range[0], self.freq_range[1], self.step)), *X_test.shape), dtype=np.float64)
            i = 0
            for f in range(self.freq_range[0], self.freq_range[1], self.step):            
                filt_X_test[i, :, :, :] = mne.filter.filter_data(X_test, self.s_freq, l_freq=f, h_freq=f+self.width)
                i += 1

            # Spatial filtering
            saved_fbcsp = pickle.load(open(checkpoint_file, 'rb'))    

            y_classes_unique = np.unique(y_test)
            pred_y_test = np.zeros((y_test.shape[0], self.class_num), dtype=np.float)
            for cls_of_interest in range(self.class_num):
                checkpoint_file_c = checkpoint_file + f'_class_{cls_of_interest}'
                
                trans_X_test = saved_fbcsp.transform(filt_X_test, class_idx=cls_of_interest)   
                
                saved_clf = pickle.load(open(checkpoint_file_c, 'rb'))
                
                pred_y_test[:,cls_of_interest] = saved_clf.predict(trans_X_test)
            
            pred_y_test_multi = np.asarray([np.argmin(pred_y_test[i,:]) for i in range(pred_y_test.shape[0])])  
            preds_cands.append(pred_y_test_multi) 
        
        final_idxs = [] # (# of trials, )
        for n in range(preds_cands[0].shape[0]):
            temp_preds_cands = []
            for m in range(len(preds_cands)):
                pred_y = preds_cands[m][n]
                temp_preds_cands.append(pred_y)           
            c = Counter(temp_preds_cands) # {value: counted number} starting with the highest counted number value
            final_idx = c.most_common()[0][0] 
            final_idxs.append(final_idx)
            
        accs.update(accuracy_score(final_idxs, y_test), inputs.size(0))

        dur_time = time.time() - start_time
        times.update(dur_time)
        accs.update(accuracy_score(final_idxs, y_test), inputs.size(0))
        return accs.avg          
        