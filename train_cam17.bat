@echo off
FOR /f "usebackq" %%i IN (`PowerShell ^(Get-Date ^).ToString^('yyyyMMdd_HHmm'^)`) DO SET DTime=%%i

SET OUTPUT_FOLDER=training/%DTime%_cpc_test


python -m GreedyInfoMax.vision.main_vision ^
--data_input_dir "E:/data/camelyon_tumor_20x" ^
--dataset "cam17" ^
--batch_size 2 ^
--validate ^
--model_splits 1 ^
--num_epochs 2 ^
--save_dir %OUTPUT_FOLDER% ^
--negative_samples 5 ^
--resnet 34 ^
--output_dim 32 ^
--infoloss_acc ^
--pred_directions 2


