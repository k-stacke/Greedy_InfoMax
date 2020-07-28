#!/bin/bash

DTime=$( date +%Y%m%d_%H%M )

FOLDER=$1
MODEL=$2
OUTPUT_DIM=${3:-1024}

OUTPUT_FOLDER='/proj/karst/results/'$FOLDER'/linear_classification_'$MODEL''

python -m GreedyInfoMax.vision.downstream_classification \
--data_input_dir '/proj/karst/cam17_20x' \
--dataset 'cam17' \
--batch_size 256 \
--num_epochs 10 \
--model_splits 1 \
--model_type 1 \
--validate \
--output_dim $OUTPUT_DIM \
--save_dir $OUTPUT_FOLDER \
--training_data_csv '/proj/karst/results/dataframes/supervised_training_patches_0.csv' \
--test_data_csv '/proj/karst/results/dataframes/supervised_test_patches_0.csv' \
#--model_path '/proj/karst/results/'$FOLDER'' \
#--model_num $MODEL \

# Copy results to disk
#cp -R $OUTPUT_FOLDER /mnt/tmp
