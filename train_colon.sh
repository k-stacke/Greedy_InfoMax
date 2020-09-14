#!/bin/bash

DTime=$( date +%Y%m%d_%H%M )

#OUTDIM=$1

python -m GreedyInfoMax.vision.main_vision \
--data_input_dir '/proj/karst/colon/exp_1911apka/colon_imagedata' \
--dataset 'cam17' \
--batch_size 32 \
--validate \
--model_splits 1 \
--num_epochs 50 \
--save_dir '/proj/karst/results/'$DTime'_greedyinfomax_colon0' \
--output_dim 1024 \
--training_data_csv '/home/sectra-karst/dataframes/colon/training_patches_exl_val_center0.csv' \
--test_data_csv '/home/sectra-karst/dataframes/colon/validation_patches_center0.csv' \
--pred_directions 4 \
--ten_x \
#--patch_aug
#--big_patches
#--model_path 'logs/experiment_nodali' \
#--start_epoch 14 \
#--domain_loss \
#--domain_loss_reg 0.1 \


