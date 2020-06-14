""" This script runs a linear regression based on
(a) the selected features determined a-priori
(b) the window size for the training data set determined a-priori from rolling regression and cross-validation
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from my_functions import adjust_data_type, check_for_any_nans, createdir
from MyLinearRegression import *

# Path where file is currently running
dir_path = os.path.dirname(os.path.realpath(__file__))

# -------------------------------------------------------------------------------------------------------------------- #
# Import data
features = pickle.load(open(dir_path + "/features_clean_mw.pkl", "rb"))
targets = pickle.load(open(dir_path + "/target_clean_mw.pkl", "rb"))

feature_names = features.columns.tolist()
target_names = targets.columns.tolist()

# -------------------------------------------------------------------------------------------------------------------- #
# Several possibilitis for feature engineering:

x_df = features.copy()
y_df = targets.copy()

# Add additional features
x_squared = (x_df ** 2).rename({"x0": "x0**2", "x1": "x1**2", "x2": "x2**2"}, axis=1)
x_cube = (x_df ** 3).rename({"x0": "x0**3", "x1": "x1**3", "x2": "x2**3"}, axis=1)

features_new = pd.concat([x_df, x_squared, x_cube], axis=1)

# Define train and test period: Currently, all data is used priod to last available day
t_test = x_df.last('1D').index
t_train = x_df.loc[:t_test[0] - pd.Timedelta("30T"), :].index

x_train = features_new.loc[t_train, :].values
y_train = targets.loc[t_train, :].values
x_test = features_new.loc[t_test, :].values
y_test = targets.loc[t_test, :].values

x_test_original = features_new.loc[t_test, feature_names].values

# -------------------------------------------------------------------------------------------------------------------- #
# Run regression

# Run my regression
my_regression = MyLinearRegression()
my_regression.fit(x_train, y_train)
my_pred = my_regression.predict(x_test)
my_error = error(y_test, my_pred)
my_mse_score = my_regression.score(y_test, my_pred, metric="MSE")
my_mae_score = my_regression.score(y_test, my_pred, metric="MAE")
my_weights = my_regression.weights

x_mean_error, x_mean_mse_score = my_regression.benchmark_mean(x_test_original, y_test, metric="MSE")
vendor_error, vendor_mse_score = my_regression.benchmark_vendor(x_test_original, y_test, metric="MSE")
_, x_mean_mae_score = my_regression.benchmark_mean(x_test_original, y_test, metric="MAE")
_, vendor_mae_score = my_regression.benchmark_vendor(x_test_original, y_test, metric="MAE")

