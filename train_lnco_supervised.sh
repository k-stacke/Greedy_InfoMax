#!/bin/bash

DTime=$( date +%Y%m%d_%H%M )

NUMSAMPLES=$1
COUNTER=$2

OUTPUT_FOLDER='/mnt/data/trainings/'$DTime'_GreedyLinear_COLON_'$NUMSAMPLES'_'$COUNTER''

python -m GreedyInfoMax.vision.downstream_classification \
--data_input_dir '/mnt/data/tumor' \
--dataset 'cam17' \
--batch_size 32 \
--num_epochs 10 \
--save_dir $OUTPUT_FOLDER \
--model_path 'logs/experiment_color_unbiased_dataset' \
--model_num 16 \
#--training_data_csv '/mnt/data/.csv' \
#--test_data_csv 'logs/experiment_color_unbiased_dataset/test_patches_newlabel.csv' \

# Copy results to disk
cp -R $OUTPUT_FOLDER /mnt/tmp
