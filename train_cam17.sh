#!/bin/bash

DTime=$( date +%Y%m%d_%H%M )

OUTDIM=$1

python -m GreedyInfoMax.vision.main_vision \
--data_input_dir '/proj/karst/slide_data202003' \
--dataset 'cam17' \
--batch_size 16 \
--validate \
--model_splits 1 \
--num_epochs 50 \
--save_dir '/proj/karst/results/'$DTime'_greedyinfomax' \
--output_dim $OUTDIM \
--infoloss_acc \
--training_data_csv '/proj/karst/results/dataframes/training_patches_exl_val.csv' \
--test_data_csv '/proj/karst/results/dataframes/validation_patches.csv' \
--pred_directions 4
#--patch_aug
#--big_patches
#--model_path 'logs/experiment_nodali' \
#--start_epoch 14 \
#--domain_loss \
#--domain_loss_reg 0.1 \


