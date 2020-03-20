#!/bin/bash

DTime=$( date +%Y%m%d_%H%M )


SAMPLINGMETHOD=$1
NUMSAMPLES=$2
COUNTER=$3

OUTPUT_FOLDER='/mnt/data/trainings/'$DTime'_GreedyLinear_'$SAMPLINGMETHOD'_'$NUMSAMPLES'_'$COUNTER''

python -m GreedyInfoMax.vision.downstream_classification \
--data_input_dir '/mnt/data/slide/slide_data202003' \
--dataset 'cam17' \
--batch_size 32 \
--num_epochs 10 \
--save_dir $OUTPUT_FOLDER \
--model_path 'logs/experiment_color_unbiased_dataset' \
--model_num 16 \
--training_data_csv '/mnt/data/slide/sampling_dataframes/'$COUNTER'/training_patches_'$SAMPLINGMETHOD'_'$NUMSAMPLES'.csv' \
--test_data_csv 'logs/experiment_color_unbiased_dataset/test_patches_newlabel.csv' \

# Copy results to disk
cp -R $OUTPUT_FOLDER /mnt/tmp
