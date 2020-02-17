#!/bin/bash


python -m GreedyInfoMax.vision.main_vision \
--start_epoch 1 \
--model_path './logs/experiment_cam_lnco_copy' \
--data_input_dir '/mnt/data/tumor' \
--dataset 'cam17' \
--batch_size 32 
