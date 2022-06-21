#! bin/bash/

CUDA_VISIBLE_DEVICES=0,1,2 python3 ../train.py --t_min .5 --model_type FBCSP --batch_size 1

