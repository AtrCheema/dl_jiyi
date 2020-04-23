Code to predict ARGs and OTUs at beach using Neural Network (mainly LSTM).

This repo expects that a folder named `data` exists. This must either contain all the input data as one `.xlsx` file or a sub-folder inside `data` named `AWS_data` which
contains different un-processed data for two sites.

The folder `busan` must contain `.txt` files which must contain following header/columns names: `Date_Time	tide_cm	wat_temp_c	sal_psu	wind_speed_mps	wind_dir_d	air_temp_c	air_p_hpa`.

Target OTUs, ARGs and E. Coli data must be present in folder `data` in a file named `Time_series_data.xlsx`. 

Use `run_model.py` file change hyper-parameters of model.
