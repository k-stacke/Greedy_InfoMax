#!/bin/bash

DTime=$( date +%Y%m%d_%H%M )

#OUTDIM=$1

python -m GreedyInfoMax.vision.main_vision \
--data_input_dir '/proj/karst/camelyon_complete' \
--dataset 'cam17' \
--batch_size 32 \
--validate \
--model_splits 1 \
--num_epochs 50 \
--save_dir '/proj/karst/results/'$DTime'_greedyinfomax_scanner0' \
--output_dim 1024 \
--training_data_csv '/home/sectra-karst/dataframes/camelyon_complete/training_patches_exl_val_scanner0.csv' \
--test_data_csv '/home/sectra-karst/dataframes/camelyon_complete/validation_patches_scanner0.csv' \
--pred_directions 4 \
--ten_x
#--patch_aug
#--big_patches
#--model_path 'logs/experiment_nodali' \
#--start_epoch 14 \
#--domain_loss \
#--domain_loss_reg 0.1 \


