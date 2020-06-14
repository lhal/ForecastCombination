# ForecastCombination
This project implements a mutlivariate linear regression model to improve wind power forecast combinations and is benchmarked with (a) the mean forecast of the 3 vendor forecasts and (b) the best single vendor forecast. 

The regression model is implemented as a class in MyLinearRegression.py

The data set is cleaned in data_cleaning.py and the clean data are stored in "target_clean_mw.pkl" and "features_clean_mw.pkl".

Feature selection is studied in model_selection.py, along with a setup to perform rolling regressions and cross-validation to estimate the models performance over time and with different training sizes. 

Once, appropriate features are selected and the optimal training size is determined a linear regression can be run in run_regression.py (currently, it uses the square and the cube of features as additional features). 

run_regression.py relies on clean and pre-processed data in the form of "target_clean_mw.pkl" and "features_clean_mw.pkl".
