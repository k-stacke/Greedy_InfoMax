#!/bin/bash


python -m GreedyInfoMax.vision.main_vision \
--data_input_dir '/mnt/data/slide/slide_data202003' \
--dataset 'cam17' \
--batch_size 32 \
--save_dir 'experiment_test_merge' \
--training_data_csv 'logs/experiment_test_merge/training_patches_exl_val_newlabel.csv' \
--test_data_csv 'logs/experiment_test_merge/test_patches_newlabel.csv' \
#--model_path 'logs/experiment_color_w_domain' \
#--start_epoch 0 \
#--domain_loss \
#--domain_loss_reg 0.1 \


