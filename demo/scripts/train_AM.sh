#! bin/bash/

CUDA_VISIBLE_DEVICES=0,1,2 python3 ../train.py --is_agent_module True --batch_size 230
