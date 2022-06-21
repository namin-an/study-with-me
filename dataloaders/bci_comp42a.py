# References
# - http://github.com/talhaanwarch/youtube-tutorials/BCI_Competition_IV.ipynb
# - https://programs.wiki/wiki/record-the-bci-competition-iv-2a-process-of-the-first-ann-run.html

import os
import argparse
import warnings
from glob import glob
from tqdm import tqdm
warnings.simplefilter("ignore", DeprecationWarning)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mne
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
import scipy.io
from braindecode.datautil.preprocess import exponential_moving_standardize


class BCIComp42aDataLoader(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path, is_test, target_subject, s_freq, l_freq, h_freq, t_min, t_max):
        """
        'reject' (1023): 1
        'eye move' (1072): 2
        'eye open' (276): 3
        'eye close' (277): 4
        'new run' (32766): 5
        'new trial' (768): 6
        'class 1' (769): 7 -> 0
        'class 2' (770): 8 -> 1
        'class 3' (771): 9 -> 2
        'class 4' (772): 10 -> 3
        'class unknown' (783): 7 -> 0
        """

        self.data_path = data_path
        self.label_path = label_path
        self.is_test = is_test
        self.s_freq = s_freq
        self.l_freq, self.h_freq = l_freq, h_freq
        self.t_min, self.t_max = t_min, t_max
        self.subject_num = target_subject
        
        self.features, self.labels = self.concatenate_data()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):   
        return {'features' : self.features[idx, ...],
                'labels' : self.labels[idx]}
        
    def read_data(self, i, file_names, label_names):
        raw = mne.io.read_raw_gdf(file_names, preload=True, eog=['EOG-left', 'EOG-central', 'EOG-right'], verbose=False)
        raw = raw.resample(sfreq=self.s_freq) # downsampling 
        events, event_ids = mne.events_from_annotations(raw)

        raw.filter(self.l_freq, self.h_freq, fir_design='firwin') # band pass filter
        raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']

        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                               exclude='bads')

        if not self.is_test:
          if i != 3:
            event_id = {'769': 7, '770': 8, '771': 9, '772': 10}
          else:
            event_id = {'769': 5, '770': 6, '771': 7, '772': 8}
        else:
          event_id = {'783': 7}

        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=self.t_min, tmax=self.t_max,
                            proj=True, picks=picks, baseline=None, preload=True, on_missing='warn') # baseline=None for when t_min=0
        features = epochs.get_data() * 1e6 # convert V to uV (1 V = 1e6 uV)
        normalized_features = [exponential_moving_standardize(feature, factor_new=0.001, init_block_size=int(raw.info['sfreq'] * 4)) for feature in features]
        labels = scipy.io.loadmat(label_names)['classlabel'] - 1 # start from 0
                                                              
        return normalized_features, labels, epochs

    def concatenate_data(self):
        if not self.is_test:
            file_list = sorted(glob(f'{self.data_path}/*T.gdf'))
            label_list = sorted(glob(f'{self.label_path}/*T.mat'))
        else:
            file_list = sorted(glob(f'{self.data_path}/*E.gdf'))
            label_list = sorted(glob(f'{self.label_path}/*E.mat'))       
        assert len(file_list) == len(label_list)
        
        i = self.subject_num
        file_name, label_name = file_list[i], label_list[i] # (288 trials, C, T), (288 trials, )
        features, labels, epochs = self.read_data(i, file_name, label_name) 
                
        features = np.expand_dims(features, axis=1) # channel expansion

        return features, labels

