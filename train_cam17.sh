#!/bin/bash


python -m GreedyInfoMax.vision.main_vision \
--data_input_dir '/mnt/data/slide/slide_data202003' \
--dataset 'cam17' \
--batch_size 32 \
--validate \
--save_dir 'experiment_color_unbiased_dataset' \
#--model_path 'logs/experiment_nodali' \
#--start_epoch 14 \
#--training_data_csv 'logs/experiment_color_nodali/training_patches.csv' \
#--test_data_csv 'logs/experiment_color_nodali/test_patches.csv' \
#--domain_loss \
#--domain_loss_reg 0.1 \


