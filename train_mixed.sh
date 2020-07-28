#!/bin/bash

DTime=$( date +%Y%m%d_%H%M )

#OUTDIM=$1

python -m GreedyInfoMax.vision.main_vision \
--data_input_dir '/proj/karst' \
--dataset 'cam17' \
--batch_size 32 \
--validate \
--model_splits 1 \
--num_epochs 70 \
--save_dir '/proj/karst/results/'$DTime'_greedyinfomax_mixed' \
--output_dim 1024 \
--training_data_csv '/proj/karst/results/dataframes/combined_unsupervised/training_patches_exl_val.csv' \
--test_data_csv '/proj/karst/results/dataframes/combined_unsupervised/validation_patches.csv' \
--pred_directions 4 
#--patch_aug
#--big_patches
#--model_path 'logs/experiment_nodali' \
#--start_epoch 14 \
#--domain_loss \
#--domain_loss_reg 0.1 \


