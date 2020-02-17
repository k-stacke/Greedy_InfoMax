!#/bin/sh


python -m GreedyInfoMax.vision.downstream_classification \
--data_input_dir '/mnt/data/camelyon_20x' \
--dataset cam17 \
--model_path ./logs/vision_experiment \
--model_num 12 \
num_epochs 1
