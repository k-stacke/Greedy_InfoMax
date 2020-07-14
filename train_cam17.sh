#!/bin/bash

DTime=$( date +%Y%m%d_%H%M )
python -m GreedyInfoMax.vision.main_vision \
--data_input_dir '/proj/karst/slide_data202003' \
--dataset 'cam17' \
--batch_size 32 \
--validate \
--model_splits 1 \
--num_epochs 100 \
--save_dir '/proj/karst/results/'$DTime'_greedyinfomax' \
#--training_data_csv 'logs/test/training_patches.csv' \
#--test_data_csv 'logs/test/test_patches.csv' \
#--model_path 'logs/experiment_nodali' \
#--start_epoch 14 \
#--domain_loss \
#--domain_loss_reg 0.1 \


