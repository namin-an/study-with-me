# Reference:
# - https://github.com/fbcsptoolbox/fbcsp_code/blob/master/bin/MLEngine.py
# - https://github.com/ADD-Drone/drone-sjkim/blob/master/models/backbones/FBCSP.py
# - https://github.com/stupiddogger/FBCSP/blob/master/FBCSP.py

import time
import numpy as np
import pickle
import torch
import mne
from sklearn.svm import SVR
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch.utils.tensorboard import SummaryWriter
    
from utils.average_meter import AverageMeter
from FBCSP.FBCSP import FBCSP
    
class ExperimentFBCSP:
    def __init__(self, class_num, s_freq, l_freq, h_freq, width, step, m_filters:int=2):
        self.class_num = class_num
        self.s_freq = s_freq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.freq_range = [l_freq, h_freq]
        self.width = width
        self.step = step
        self.m_filters = m_filters 
        
    def train(self, train_dataloader, valid_dataloader, checkpoint_file):
        times = AverageMeter()
        
        # Iterate train data loader
        X_train, y_train = np.array([]), np.array([])
        for data in train_dataloader:
            inputs = data['features']
            targets = data['labels']
            
            if X_train.shape[0] == 0 and y_train.shape[0] == 0:
                X_train, y_train = inputs, targets
            else:
                X_train = np.concatenate([X_train, inputs], axis=0)
                y_train = np.concatenate([y_train, targets], axis=0)    
        X_train, y_train = X_train.squeeze(), y_train.squeeze()
        
        # Iterate validation data loader
        X_valid, y_valid = np.array([]), np.array([])
        for data in valid_dataloader:
            inputs = data['features']
            targets = data['labels']
            
            if X_valid.shape[0] == 0 and y_valid.shape[0] == 0:
                X_valid, y_valid = inputs, targets
            else:
                X_valid = np.concatenate([X_valid, inputs], axis=0)
                y_valid = np.concatenate([y_valid, targets], axis=0)    
        X_valid, y_valid = X_valid.squeeze(), y_valid.squeeze()

        start_time = time.time()
        # Bandpass filtering (4~8, 8~12, ..., 36~40) (6-10, 10-14, ..., 34-38 added for over-lapping condition)         
        filt_X_train = np.zeros((len(range(self.freq_range[0], self.freq_range[1], self.step)), *X_train.shape), dtype=np.float64)
        filt_X_valid = np.zeros((len(range(self.freq_range[0], self.freq_range[1], self.step)), *X_valid.shape), dtype=np.float64)
        i = 0
        for f in range(self.freq_range[0], self.freq_range[1], self.step):  
            filt_X_train[i, :, :, :] = mne.filter.filter_data(X_train, self.s_freq, l_freq=f, h_freq=f+self.width)
            filt_X_valid[i, :, :, :] = mne.filter.filter_data(X_valid, self.s_freq, l_freq=f, h_freq=f+self.width) 
            i += 1
                
        # Spatial filtering
        fbcsp = FBCSP(self.m_filters)
        fbcsp.fit(filt_X_train, y_train)
        pickle.dump(fbcsp, open(checkpoint_file, 'wb'))            
        
        y_classes_unique = np.unique(y_train)
        pred_y_train = np.zeros((y_train.shape[0], self.class_num), dtype=np.float)
        pred_y_valid = np.zeros((y_valid.shape[0], self.class_num), dtype=np.float)
        for cls_of_interest in range(self.class_num):
            checkpoint_file_c = checkpoint_file + f'_class_{cls_of_interest}'
            
            select_class_labels = lambda c, y_labels: [0 if y == c else 1 for y in y_labels]
            y_train_cls = np.asarray(select_class_labels(cls_of_interest, y_train))
            trans_X_train = fbcsp.transform(filt_X_train, class_idx=cls_of_interest)
            trans_X_valid = fbcsp.transform(filt_X_valid, class_idx=cls_of_interest)
            
            clf = SVR(gamma='auto') 
            # clf = LinearDiscriminantAnalysis()
            clf.fit(trans_X_train, np.asarray(y_train_cls, dtype=np.float))
            pickle.dump(clf, open(checkpoint_file_c, 'wb'))

            pred_y_train[:,cls_of_interest] = clf.predict(trans_X_train)
            pred_y_valid[:,cls_of_interest] = clf.predict(trans_X_valid)

        pred_y_train_multi = np.asarray([np.argmin(pred_y_train[i,:]) for i in range(pred_y_train.shape[0])])  
        pred_y_valid_multi = np.asarray([np.argmin(pred_y_valid[i,:]) for i in range(pred_y_valid.shape[0])])  
        acc = np.sum(pred_y_train_multi == y_train, dtype=np.float) / len(y_train)
        valid_acc = np.sum(pred_y_valid_multi == y_valid, dtype=np.float) / len(y_valid)
             
        dur_time = time.time() - start_time
        times.update(dur_time)
        return valid_acc

    
   