#!/bin/bash


python -m GreedyInfoMax.vision.main_vision \
--data_input_dir '/mnt/data/tumor' \
--dataset 'cam17' \
--batch_size 42 \
--domain_loss \
--domain_loss_reg 0.1 \
--save_dir 'experiment_color_w_domain' \
#--model_path 'logs/experiment_color_w_domain' \
#--start_epoch 0 \
#--training_data_csv 'logs/experiment_color_w_domain/training_patches.csv' \
#--test_data_csv 'logs/experiment_color_w_domain/test_patches.csv'


