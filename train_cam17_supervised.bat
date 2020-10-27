@echo off
FOR /f "usebackq" %%i IN (`PowerShell ^(Get-Date ^).ToString^('yyyyMMdd_HHmm'^)`) DO SET DTime=%%i


SET MODEL=%1

SET OUTPUT_FOLDER="E:/OneDrive - Sectra/Research/2020/cpc/20200713_2319_greedyinfomax/linear_classification_%MODEL%"

SET MODEL_PATH="E:/OneDrive - Sectra/Research/2020/cpc/20200713_2319_greedyinfomax"
SET DATAFRAME_PATH="E:/OneDrive - Sectra/Research/2020/cpc/dataframes"

python -m GreedyInfoMax.vision.downstream_classification ^
--data_input_dir "F:/data/camelyon17/slide_data202003" ^
--dataset "cam17" ^
--batch_size 12 ^
--num_epochs 10 ^
--model_splits 1 ^
--model_type 0 ^
--validate ^
--save_dir %OUTPUT_FOLDER% ^
--model_path %MODEL_PATH%/models ^
--model_num %MODEL% ^
--training_data_csv %DATAFRAME_PATH%/supervised_training_patches_1.csv ^
--test_data_csv %DATAFRAME_PATH%/supervised_test_patches_1.csv ^


