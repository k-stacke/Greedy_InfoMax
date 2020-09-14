#!/bin/sh
DTime=$( date +%Y%m%d_%H%M )

FOLDER=$1
MODEL=$2

#FOLDER=''$DTime'_greedyinfomax'
#OUTPUT_FOLDER='/proj/karst/results/'$FOLDER'/stl10/linear_classification_'$MODEL''

echo "Training the Greedy InfoMax Model on vision data (stl-10)"
python -m GreedyInfoMax.vision.main_vision \
--dataset 'svhn' \
--download_dataset \
--batch_size 32 \
--model_splits 1 \
--num_epochs 100 \
--save_dir '/proj/karst/results/'$DTime'_greedyinfomax_svhn' \
--output_dim 1024 \
--pred_directions 1 \


#echo "Testing the Greedy InfoMax Model for image classification"
#python -m GreedyInfoMax.vision.downstream_classification \
#--model_num $MODEL \
#--batch_size 256 \
#--num_epochs 50 \
#--model_splits 1 \
#--model_type 0 \
#--validate \
#--save_dir ''$OUTPUT_FOLDER'/dataset_0' \
#--model_path '/proj/karst/results/'$FOLDER'' \
