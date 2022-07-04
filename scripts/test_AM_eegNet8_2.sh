#! bin/bash/

CUDA_VISIBLE_DEVICES=0,1,2 python3 ../test.py --is_agent_module True --model_type eegNet --F1 8

