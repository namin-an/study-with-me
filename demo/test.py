# Note: You can test both SD and SI for BCI competition 4 2a dataset using the same checkpoint files created with train.py. However, since there is no separate test dataset for Drone dataset, you can only test SI here, and SD results can be checked using average of validation accuracy in training.

import os, sys
sys.path.append('/opt/pytorch/')
import argparse
import time
from datetime import datetime
import pandas as pd

import torch
import torch.nn as nn
from dataloaders.bci_comp42a import BCIComp42aDataLoader
from dataloaders.drone import DroneDataLoader

from deepshallowconvnet.deepConvNet import DeepConvNet
from deepshallowconvnet.shallowConvNet import ShallowConvNet
from eegnet.eegNet import EEGNet
from MSNN.MSNN import MSNN
from FBCSP.FBCSP import FBCSP

from utils.evaluate import Evaluate
from utils.evaluate_FBCSP import EvaluateFBCSP

if __name__ == '__main__':
    device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
    print(torch.cuda.device_count(), device, 'devices available')
    today = datetime.now().strftime('%Y%m%d')
    today = '20220627'
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--is_agent_module', type=bool, default=False)
    
    parser.add_argument('--is_subject_independent', type=bool, default=False)
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
    
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--model_type', type=str, default='shallowConvNet')
    parser.add_argument('--F1', type=int, default=4)
    # parser.add_argument('--input_shape', type=int, nargs='+', default=[1, 1, 22, 500])
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    
    args = parser.parse_args()
    
    # Name file and path names.
    if args.model_type == 'eegNet':
        model_name = args.model_type + str(args.F1) + '_2'
    else:
        model_name = args.model_type
    if args.data_type == 'bci_comp4a' or args.data_type == 'drone':
        class_num = 4
    if args.is_agent_module:
        extra_name = '_agents'
    else:
        extra_name = ''
        
    subject_dict = {'Subject #':[], 'Accuracy':[]}
    for subject_num in range(args.tot_subject_num): 
        print(f"\nSubject: {subject_num}")
        
        # Check if there exist saved models.
        checkpoint_files = []       
        # Subject-dependent scenario 
        if not args.is_subject_independent and not args.data_type == 'drone':
            for fold in range(args.fold_num):
                checkpoint_file = f'../checkpoints/{today}/{args.data_type}/{model_name}_subject{subject_num}_fold{fold}{extra_name}'
                if os.path.isfile(checkpoint_file):
                    checkpoint_files.append(checkpoint_file)     
        # Subject-independent scenario
        else:
            for fold in range(args.fold_num):
                for other_subject_num in range(args.tot_subject_num):
                    if other_subject_num == subject_num: continue # except for target subject
                    checkpoint_file = f'../checkpoints/{today}/{args.data_type}/{model_name}_subject{other_subject_num}_fold{fold}{extra_name}'
                    if os.path.isfile(checkpoint_file):
                        checkpoint_files.append(checkpoint_file) 

        if len(checkpoint_files) > 0:            
            # Load test data.
            if args.data_type == 'bci_comp4a':
                dataset = BCIComp42aDataLoader(data_path=args.data_path,
                                               label_path=args.label_path,
                                               is_test=True,
                                               target_subject = subject_num, # subject #0 ~ #8
                                               s_freq=args.s_freq,
                                               l_freq=args.l_freq, h_freq=args.h_freq,
                                               t_min=args.t_min, t_max=args.t_max)
            elif args.data_type == 'drone':
                dataset = DroneDataLoader(data_path=args.data_path,
                                          task='MI',
                                          is_test=True,
                                          target_subject = subject_num, # subject #0 ~ #24
                                          s_freq=args.s_freq, # 500 Hz -> 100 Hz
                                          l_freq=args.l_freq, h_freq=args.h_freq,
                                          t_min=args.t_min, t_max=args.t_max)
            test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4*torch.cuda.device_count())

            # Prepare chosen model.
            test_features, test_label = next(iter(test_dataloader))['features'], next(iter(test_dataloader))['labels']
            print(f"shape of features: {test_features.shape}")
            print(f"shape of label: {test_label.shape}")
            X = torch.rand(test_features.shape).to(device)
            if args.model_type == 'deepConvNet':
                net = DeepConvNet(test_features.shape[1:], args.s_freq, class_num).to(device)
            elif args.model_type == 'shallowConvNet':
                net = ShallowConvNet(test_features.shape[1:], args.s_freq, class_num).to(device)
            elif args.model_type == 'mshallowConvNet':
                    net = MShallowConvNet(test_features.shape[1:], args.s_freq, class_num).to(device)
            elif args.model_type == 'eegNet':
                net = EEGNet(test_features.shape[1:], args.s_freq,class_num, args.dropout_rate, args.F1).to(device)                   
            elif args.model_type == 'MSNN':
                net = MSNN(test_features.shape[1:], args.s_freq, class_num, args.dropout_rate).to(device)                   
            elif args.model_type == 'FBCSP':
                net = EvaluateFBCSP(class_num, args.s_freq, args.l_freq, args.h_freq, args.width, args.step, args.m_filters)
            if args.model_type != 'FBCSP':
                y = net(X) # initialize model parameters for Dataparallel and lazy modules
                print(f"shape of final output: {y.shape}")
                net = net.to(device)
                
                # Test saved neural network.
                net.eval()
                exp = Evaluate(test_dataloader, net, device, checkpoint_files)
                result = exp.test()
                result = result.item()
            elif args.model_type == 'FBCSP':
                result = net.test(test_dataloader, checkpoint_files)
            # Print out the final test accuracy
            subject_dict['Subject #'].append(subject_num+1)
            subject_dict['Accuracy'].append(round(result, 4))
            
    subject_df = pd.DataFrame.from_dict(subject_dict)
    subject_df.columns = ['Subject #', 'Accuracy']
    print(subject_df)
    print(subject_df['Accuracy'].mean())
 