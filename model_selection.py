""" This script evaluates
(a) which features to consider in the forecast combination
(b) the best window size for the training data set and evaluate the impact of seasonal changes

(1) Rolling window (coefficient might change over time)
(2) Cross-validation: Vary window size in rolling
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from my_functions import adjust_data_type, check_for_any_nans, createdir
from MyLinearRegression import *

data_path = "/home/lejla/Documents/ForecastCombination/CleanedData/"

# -------------------------------------------------------------------------------------------------------------------- #
# Import data
features = pickle.load(open(data_path + "features_clean_mw.pkl", "rb"))
targets = pickle.load(open(data_path + "target_clean_mw.pkl", "rb"))

feature_names = features.columns.tolist()
target_names = targets.columns.tolist()


# -------------------------------------------------------------------------------------------------------------------- #
# Several possibilitis for feature engineering:
# (1) add x1**2, x2**2, x3**2 (forecast performance can depend on power level)
# (2) add x1(t-1), x2(t-1), x3(t-1) (time dependency)

x_df = features.copy()
y_df = targets.copy()

# Test again with all data except for last day
t_test = x_df.last('1D').index
t_train = x_df.loc[:t_test[0] - pd.Timedelta("30T"), :].index

# Define feature sets
x_sqr_df = pd.concat([x_df, (x_df ** 2).rename({"x0": "x0**2", "x1": "x1**2", "x2": "x2**2"}, axis=1)], axis=1)
x_cube_df = pd.concat([x_df,
                       (x_df ** 2).rename({"x0": "x0**2", "x1": "x1**2", "x2": "x2**2"}, axis=1),
                       (x_df ** 3).rename({"x0": "x0**3", "x1": "x1**3", "x2": "x2**3"}, axis=1)], axis=1)
x_dfs_names = ["Original features", "Original + squared", "Original + cube"]
x_dfs = [x_df, x_sqr_df, x_cube_df]

features_result_keys = ["my_pred", "my_error", "my_mse_score", "my_mae_score", "my_weights", "x_mean_error",
                        "x_mean_mse_score", "x_mean_mae_score", "vendor_error", "vendor_mse_score", "vendor_mae_score"]
feature_results = {key: [] for key in features_result_keys}
for feature_set in x_dfs:
    x_train = feature_set.loc[t_train, :].values
    y_train = targets.loc[t_train, :].values

    x_test = feature_set.loc[t_test, :].values
    y_test = targets.loc[t_test, :].values

    x_test_original = feature_set.loc[t_test, feature_names].values

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

    _res = [my_pred, my_error, my_mse_score, my_mae_score, my_weights, x_mean_error, x_mean_mse_score, x_mean_mae_score,
            vendor_error, vendor_mse_score, vendor_mae_score]

    for idx, i in enumerate(features_result_keys):
        feature_results[i].append(_res[idx])

# -------------------------------------------------------------------------------------------------------------------- #
# Evaluate results of feature selection

save_path = createdir("/home/lejla/Documents/ForecastCombination/ModelComparison/")

_dfs_1 = []
_dfs_2 = []
_dfs_3 = []
for key in ["my_error", "x_mean_error"]:
    err = np.concatenate(feature_results[key], axis=1)
    _dfs_1.append(pd.DataFrame(err[:, 0], index=t_test, columns=[key]))
    _dfs_2.append(pd.DataFrame(err[:, 1], index=t_test, columns=[key]))
    _dfs_3.append(pd.DataFrame(err[:, 2], index=t_test, columns=[key]))

errors = [pd.concat(_dfs_1, axis=1), pd.concat(_dfs_2, axis=1), pd.concat(_dfs_3, axis=1)]

for error_idx, error in enumerate(errors):
    model_name = x_dfs_names[error_idx]

    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    error.plot(ax=ax)
    ax.set_ylabel("Error (MW)")
    ax.set_title(f"Error for model {model_name}")
    plt.tight_layout()
    plt.savefig(save_path + f"00_error_model_{error_idx}.png", dpi=120)

mse_scores = []
for key in ["my_mse_score", "x_mean_mse_score"]:
    mse_sc = np.concatenate(feature_results[key], axis=0)
    mse_scores.append(pd.DataFrame(mse_sc, index=x_dfs_names, columns=[key]))

mse_scores = pd.concat(mse_scores, axis=1)

plt.close("all")
fig, ax = plt.subplots(1, 1)
mse_scores.plot.bar(ax=ax)
ax.set_ylabel("MSE")
ax.set_title(f"MSE for all three models")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(save_path + f"01_mse_score_comparison.png", dpi=120)

mae_scores = []
for key in ["my_mae_score", "x_mean_mae_score"]:
    mae_sc = np.concatenate(feature_results[key], axis=0)
    mae_scores.append(pd.DataFrame(mae_sc, index=x_dfs_names, columns=[key]))

mae_scores = pd.concat(mae_scores, axis=1)

plt.close("all")
fig, ax = plt.subplots(1, 1)
mae_scores.plot.bar(ax=ax)
ax.set_ylabel("MAE")
ax.set_title(f"MAE for all three models")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(save_path + f"01_mae_score_comparison.png", dpi=120)

# -------------------------------------------------------------------------------------------------------------------- #
# Rolling window

test_size = "24H"
window_sizes = ["30D", "60D", "90D", "120D"]
result_keys = ["training_windows", "test_windows", "my_error", "my_mse_score", "my_mae_score",
               "mean_error", "mean_mse_score", "mean_mae_score", "vendor_error", "vendor_mse_score",
               "vendor_mae_score"]

cross_validation_results = []
for window_size in window_sizes:
    train_windows, test_windows = sliding_windows(x_df, window_size=window_size, test_size=test_size)

    result_dict = {key: [] for key in result_keys}
    result_dict["training_windows"] = train_windows
    result_dict["training_windows"] = test_windows

    for t_idx, (t_train, t_test) in enumerate(zip(train_windows, test_windows)):
        print('\rTraining/testing phase ' + str(t_idx), end=' ')

        x_train = x_df.loc[t_train, :].values
        y_train = y_df.loc[t_train, :].values

        x_test = x_df.loc[t_test, :].values
        y_test = y_df.loc[t_test, :].values

        x_vendors = x_df.loc[t_test, feature_names].values

        # Run my regression
        my_regression = MyLinearRegression()
        my_regression.fit(x_train, y_train)
        my_pred = my_regression.predict(x_test)
        my_error = error(y_test, my_pred)
        my_mse_score = my_regression.score(y_test, my_pred, metric="MSE")
        my_mae_score = my_regression.score(y_test, my_pred, metric="MAE")
        my_weights = my_regression.weights

        x_mean_error, x_mean_mse_score = my_regression.benchmark_mean(x_vendors, y_test, metric="MSE")
        vendor_error, vendor_mse_score = my_regression.benchmark_vendor(x_vendors, y_test, metric="MSE")
        _, x_mean_mae_score = my_regression.benchmark_mean(x_vendors, y_test, metric="MAE")
        _, vendor_mae_score = my_regression.benchmark_vendor(x_vendors, y_test, metric="MAE")

        result_dict["my_error"].append(pd.DataFrame(my_error, index=t_test, columns=[window_size]))
        result_dict["my_mse_score"].append(pd.DataFrame(my_mse_score, index=[t_test.date[0]], columns=[window_size]))
        result_dict["my_mae_score"].append(pd.DataFrame(my_mae_score, index=[t_test.date[0]], columns=[window_size]))

        result_dict["mean_error"].append(pd.DataFrame(x_mean_error, index=t_test, columns=[window_size]))
        result_dict["mean_mse_score"].append(pd.DataFrame(x_mean_mse_score, index=[t_test.date[0]], columns=[window_size]))
        result_dict["mean_mae_score"].append(pd.DataFrame(x_mean_mae_score, index=[t_test.date[0]], columns=[window_size]))

        vendor_error_df = pd.concat({window_size: pd.DataFrame(vendor_error, index=t_test)}.values(), keys=[window_size])
        vendor_mse_score_df = pd.concat({window_size: pd.DataFrame(np.array([vendor_mse_score]),
                                                                   index=[t_test.date[0]])}.values(), keys=[window_size])
        vendor_mae_score_df = pd.concat({window_size: pd.DataFrame(np.array([vendor_mae_score]),
                                                                   index=[t_test.date[0]])}.values(), keys=[window_size])
        result_dict["vendor_error"].append(vendor_error_df)
        result_dict["vendor_mse_score"].append(vendor_mse_score_df)
        result_dict["vendor_mae_score"].append(vendor_mae_score_df)

    cross_validation_results.append(result_dict)

# -------------------------------------------------------------------------------------------------------------------- #
# Merge results and evalute -- TO BE FINISHED

for res_dict in cross_validation_results:
    for key in res_dict.keys():
        if key not in ["training_windows", "test_windows"]:
            _df = pd.concat(res_dict[key])


