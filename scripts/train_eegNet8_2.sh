#! bin/bash/

CUDA_VISIBLE_DEVICES=0,1,2 python3 ../train.py --model_type eegNet --F1 8

