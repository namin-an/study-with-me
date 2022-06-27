# Note: You can train artificial neural networks using utils.experiment/Experiment. However, FBCSP can be trained by calling .fit().

import os, sys
sys.path.append('/opt/pytorch/')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
import time
from datetime import datetime
import pandas as pd

import torch
import torch.nn as nn
from torchsummary import summary
from sklearn.model_selection import KFold

import dataloaders
from dataloaders.bci_comp42a import BCIComp42aDataLoader
from dataloaders.drone import DroneDataLoader

from deepshallowconvnet.deepConvNet import DeepConvNet
from deepshallowconvnet.shallowConvNet import ShallowConvNet
from eegnet.eegNet import EEGNet
from MSNN.MSNN import MSNN
from FBCSP.FBCSP import FBCSP

from utils.experiment import Experiment
from utils.experiment_FBCSP import ExperimentFBCSP

if __name__ == '__main__':
    device0 = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device1 = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    device2 = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
    today = datetime.now().strftime('%Y%m%d')
    today = '20220627'
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--is_agent_module', type=bool, default=False)
    
    parser.add_argument('--is_subject_independent', type=bool, default=True)
    parser.add_argument('--data_type', type=str, default='bci_comp4a')
    parser.add_argument('--fold_num', type=int, default=5)
    parser.add_argument('--tot_subject_num', type=int, default=9)
    parser.add_argument('--data_path', default='/opt/pytorch/datasets/BCIComp42a')
    parser.add_argument('--label_path', default='/opt/pytorch/datasets/BCIComp42a/true_labels')
    parser.add_argument('--visualize', type=bool, default=False)
    parser.add_argument('--s_freq', type=int, default=128)
    parser.add_argument('--l_freq', type=int, default=0)
    parser.add_argument('--h_freq', type=int, default=38)
    parser.add_argument('--width', type=int, default=4)
    parser.add_argument('--step', type=int, default=2)
    parser.add_argument('--t_min', type=float, default=-0.5)
    parser.add_argument('--t_max', type=float, default=2.5)
    parser.add_argument('--m_filters', type=int, default=2)
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_type', type=str, default='shallowConvNet')
    parser.add_argument('--F1', type=int, default=4)
    # parser.add_argument('--input_shape', type=int, nargs='+', default=[1, 1, 22, 500])
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=200)
    
    parser.add_argument('--num_epochs_pre', type=int, default=100)
    parser.add_argument('--window_size', type=int, default=4)
    parser.add_argument('--num_actions', type=int, default=2)
    parser.add_argument('--epsilon', type=int, default=1)
    parser.add_argument('--mask_path', default='/opt/pytorch/demo/masks/')
    
    args = parser.parse_args()
    
    if args.data_type == 'bci_comp4a' or args.data_type == 'drone':
        class_num = 4
    subject_dict = {'Subject #':[], 'Accuracy':[]}
    for subject_num in range(1): #args.tot_subject_num):
        # Load training data.
        if args.data_type == 'bci_comp4a':
            dataset = BCIComp42aDataLoader(data_path=args.data_path,
                                           label_path=args.label_path,
                                           is_test=False,
                                           target_subject = subject_num, # subject #0 ~ #8
                                           s_freq=args.s_freq, # 250 Hz -> 128 Hz
                                           l_freq=args.l_freq, h_freq=args.h_freq,
                                           t_min=args.t_min, t_max=args.t_max)
        elif args.data_type == 'drone':
            dataset = DroneDataLoader(data_path=args.data_path,
                                      task='MI',
                                      is_test=False, # only for subject-dependent scenaio # use later
                                      target_subject = subject_num, # subject #0 ~ #24
                                      s_freq=args.s_freq, # 500 Hz -> 100 Hz
                                      l_freq=args.l_freq, h_freq=args.h_freq,
                                      t_min=args.t_min, t_max=args.t_max)
        
        # Perform cross-validation
        kfold = KFold(n_splits=args.fold_num, shuffle=True)
        fold_list_temp = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"\nSubject: {subject_num} Fold: {fold}")
            
            # Name file and path names
            if args.model_type == 'eegNet':
                model_name = args.model_type + str(args.F1) + '_2'
            else:
                model_name = args.model_type
            os.makedirs(f'../checkpoints/{today}/{args.data_type}', exist_ok=True)
            os.makedirs(f'../logs/{today}/{args.data_type}', exist_ok=True)
            checkpoint_file = f'../checkpoints/{today}/{args.data_type}/{model_name}_subject{subject_num}_fold{fold}'
            log_path = f'../logs/{today}/{args.data_type}/{model_name}_subject{subject_num}_fold{fold}'
            if args.is_agent_module:
                condition = (not os.path.isfile(checkpoint_file)) or (args.is_agent_module ^ os.path.isfile(checkpoint_file + '_agents'))
            else:
                condition = not os.path.isfile(checkpoint_file)
            
            # If there is no trained file
            if condition:
                train_subsampler, valid_subsampler = torch.utils.data.SubsetRandomSampler(train_idx), torch.utils.data.SubsetRandomSampler(val_idx)
                train_dataloader, valid_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler), torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=valid_subsampler)

                # Prepare chosen model
                train_features, train_label = next(iter(train_dataloader))['features'], next(iter(train_dataloader))['labels']
                print(f"shape of features: {train_features.shape}")
                print(f"shape of label: {train_label.shape}")
                X = torch.rand(train_features.shape).to(device0)
                if args.model_type == 'deepConvNet':
                    net = DeepConvNet(train_features.shape[1:], args.s_freq, class_num).to(device0)
                elif args.model_type == 'shallowConvNet':
                    net = ShallowConvNet(train_features.shape[1:], args.s_freq, class_num).to(device0)
                elif args.model_type == 'mshallowConvNet':
                    net = MShallowConvNet(train_features.shape[1:], args.s_freq, class_num).to(device0)
                elif args.model_type == 'eegNet':
                    net = EEGNet(train_features.shape[1:], args.s_freq,class_num, args.dropout_rate, args.F1).to(device0)                   
                elif args.model_type == 'MSNN':
                    net = MSNN(train_features.shape[1:], args.s_freq, class_num, args.dropout_rate).to(device0)                   
                elif args.model_type == 'FBCSP':
                    net = ExperimentFBCSP(class_num, args.s_freq, args.l_freq, args.h_freq, args.width, args.step, args.m_filters)
                if args.model_type != 'FBCSP':
                    y = net(X) # initialize model parameters for Dataparallel and lazy modules
                    print(f"shape of final output: {y.shape}")
                    net = net.to(device0)
                    summary(net) 
                    
                    # Load saved weights
                    if os.path.isfile(checkpoint_file):
                        net.load_state_dict(torch.load(checkpoint_file))

                    # Train neural network
                    exp = Experiment(train_dataloader, valid_dataloader, net, args.learning_rate, args.num_epochs, args.num_epochs_pre, device0, device1, device2, checkpoint_file, log_path, args.model_type, class_num)
                    
                    # Train agent module
                    if args.is_agent_module:
                        # Create a mask and finethune the original network (environment)
                        exp.train_agents(args.window_size, args.num_actions, args.epsilon, subject_num, fold, args.mask_path)
                    else:
                        valid_acc = exp.train()
                        
                        # Test for drone dataset only
                        # Subject-independent scenario               
                        if args.data_type == 'drone' and args.is_subject_independent:
                            fold_list_temp.append(round(valid_accs.avg.item(), 4))
                        
                elif args.model_type == 'FBCSP':
                    valid_acc = net.train(train_dataloader, valid_dataloader, checkpoint_file)
                    fold_list_temp.append(valid_acc)         
                    
        # Average of the maximum validation accuracy over folds per subject
        if not args.is_subject_independent and args.data_type == 'drone':        
            subject_dict['Subject #'].append(subject_num+1)
            subject_dict['Accuracy'].append(sum(fold_list_temp)/len(fold_list_temp))
            print(subject_dict)
    if not args.is_subject_independent and args.data_type == 'drone':
        subject_df = pd.DataFrame.from_dict(subject_dict)
        subject_df.columns = ['Subject #', 'Accuracy']
        print(subject_df)
        print(subject_df['Accuracy'].mean())