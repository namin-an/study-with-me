#! bin/bash/

CUDA_VISIBLE_DEVICES=0,1,2 python3 ../train.py --is_agent_module True --model_type eegNet --F1 8 --num_epochs_pre 10


