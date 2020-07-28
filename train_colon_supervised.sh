#!/bin/bash

DTime=$( date +%Y%m%d_%H%M )

FOLDER=$1
MODEL=$2
OUTPUT_DIM=${3:-1024}

#FOLDER=''$DTime'_greedyinfomax'
OUTPUT_FOLDER='/proj/karst/results/'$FOLDER'/colon/linear_classification_'$MODEL''

sitecounter=0
#while [ $sitecounter -le 4 ]
#do

python -m GreedyInfoMax.vision.downstream_classification \
--data_input_dir '/proj/karst/colon/exp_apka1911/colon_imagedata' \
--dataset 'cam17' \
--batch_size 256 \
--num_epochs 10 \
--model_splits 1 \
--model_type 0 \
--validate \
--output_dim $OUTPUT_DIM \
--save_dir ''$OUTPUT_FOLDER'/dataset_'$sitecounter'' \
--training_data_csv '/proj/karst/results/dataframes/colon/supervised_training_patches.csv' \
--test_data_csv '/proj/karst/results/dataframes/colon/supervised_test_patches.csv' \
--model_path '/proj/karst/results/'$FOLDER'' \
--model_num $MODEL \
#--ten_x

#let sitecounter=sitecounter+1
#done
# Copy results to disk
#cp -R $OUTPUT_FOLDER /mnt/tmp
