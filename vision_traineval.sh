#!/bin/sh
DTime=$( date +%Y%m%d_%H%M )

echo "Training the Greedy InfoMax Model on vision data (stl-10)"
python -m GreedyInfoMax.vision.main_vision --download_dataset \
--batch_size 32 \
--model_splits 1 \
--num_epochs 100 \
--save_dir '/proj/karst/results/'$DTime'_greedyinfomax_stl10' \
--output_dim 1024 \
--pred_directions 1 \


#echo "Testing the Greedy InfoMax Model for image classification"
#python -m GreedyInfoMax.vision.downstream_classification --model_path ./logs/vision_experiment --model_num 299
