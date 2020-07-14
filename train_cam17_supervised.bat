@echo off
FOR /f "usebackq" %%i IN (`PowerShell ^(Get-Date ^).ToString^('yyyyMMdd_HHmm'^)`) DO SET DTime=%%i


SET SAMPLINGMETHOD=%1
SET NUMSAMPLES=%2
SET COUNTER=%3

SET OUTPUT_FOLDER=regression_classifier/%DTime%_GreedyLinear_%SAMPLINGMETHOD%_%NUMSAMPLES%_%COUNTER%

SET MODEL_PATH="E:/OneDrive - Sectra/Research/2020/project_dataembedding/tumor classification/experiment_color_unbiased_dataset"

python -m GreedyInfoMax.vision.downstream_classification ^
--data_input_dir "E:/data/camelyon17/slide_data202003" ^
--dataset "cam17" ^
--batch_size 12 ^
--num_epochs 15 ^
--save_dir %OUTPUT_FOLDER% ^
--model_path %MODEL_PATH%/models ^
--model_num 16 ^
--training_data_csv %MODEL_PATH%/sampling_dataframes/%COUNTER%/training_patches_%SAMPLINGMETHOD%_%NUMSAMPLES%.csv ^
--test_data_csv %MODEL_PATH%/test_patches_newlabel.csv ^


