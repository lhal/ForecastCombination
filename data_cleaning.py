""" Data exploration and cleaning """

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from my_functions import adjust_data_type, check_for_any_nans, createdir
from MyLinearRegression import MyLinearRegression, error, mean_square_error, root_mean_square_error, mean_absolute_error

data_path = "/home/lejla/Documents/ForecastCombination/"
save_path = createdir(data_path + "DataExploration2/")

# -------------------------------------------------------------------------------------------------------------------- #
# Import data
features_raw = pickle.load(open(data_path + "features.pkl", "rb"))
targets_raw = pickle.load(open(data_path + "target.pkl", "rb"))

# -------------------------------------------------------------------------------------------------------------------- #
# Clean data - step 1: Adjust data type

features = features_raw.astype("float")
targets = targets_raw.astype("float")

features_mw = features * 2  # transform to MW
targets_mw = targets * 2

features_mw.rename({0: "x0", 1: "x1", 2: "x2"}, axis=1, inplace=True)
feature_names = features_mw.columns.tolist()

targets_mw.rename({0: "y0"}, axis=1, inplace=True)
target_names = targets_mw.columns.tolist()

# Normalization - aggregated wind park capacity of portfolio not available
# features_mw = features_mw / targets_mw.max()[0]
# targets_mw = targets_mw / targets_mw.max()[0]

n_features = features_mw.shape[1]

# -------------------------------------------------------------------------------------------------------------------- #
# Explore data before any further cleaning

# Time resolution
t_res = pd.infer_freq(features_mw.index)

# Check for dupliacted time stamps
duplicated_time_stamps = (features_mw.index.nunique() == features_mw.shape[0])

start_date = min(features_mw.index.min(), targets_mw.index.min())
end_date = max(features_mw.index.max(), targets_mw.index.max())
date_range_string = start_date.strftime("%Y%m%d") + "-" + end_date.strftime("%Y%m%d")

plotting = False
if plotting:
    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    features_mw.plot(ax=ax)
    ax.legend(title="Vendor")
    ax.set_xlabel(f"")
    ax.set_ylabel("Power (MW)")
    ax.set_title("Power forecast values (raw data)")
    plt.tight_layout()
    plt.savefig(save_path + f"000_power_forecast_raw.png")

    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    targets_mw.plot(ax=ax)
    ax.set_xlabel(f"")
    ax.set_ylabel("Power (MW)")
    ax.set_title("Actual power production (raw data)")
    plt.tight_layout()
    plt.savefig(save_path + f"000_power_realization_raw.png")

    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    features_mw.mean(axis=1).plot(ax=ax, label="Mean forecast")
    targets_mw.plot(ax=ax, alpha=0.5)
    ax.legend()
    ax.set_xlabel(f"")
    ax.set_ylabel("Power (MW)")
    ax.set_title("Mean power forecast values (raw data)")
    plt.tight_layout()
    plt.savefig(save_path + f"000_power_forecast_raw_mean.png")

    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    ax.scatter(targets_mw["y0"], features_mw.mean(axis=1), s=2, alpha=0.4)
    ax.set_xlabel(f"Power realization y0 (MW)")
    ax.set_ylabel("Mean power forecast (MW)")
    ax.set_title(f"Mean power forecast vs. actual realization (raw data)\n({date_range_string})")
    plt.tight_layout()
    plt.savefig(save_path + f"000_xmean_vs_y0.png")

    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    ax.scatter(targets_mw["y0"].diff(), features_mw.mean(axis=1).diff(), s=2, alpha=0.4)
    ax.set_xlabel(f"Power realization difference y0.diff() (MW)")
    ax.set_ylabel("Mean power forecast difference x_mean.diff() (MW)")
    ax.set_title(f"Mean power forecast diff. vs. actual realization diff (raw data)\n({date_range_string})")
    plt.tight_layout()
    plt.savefig(save_path + f"000_xmean_diff_vs_y0_diff.png")

    y0_minus_xmean = targets_mw["y0"] - features_mw.mean(axis=1)
    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    ax.plot(y0_minus_xmean)
    y0_minus_xmean.rolling('30D').mean().plot(ax=ax, label="30D roll mean")
    ax.set_ylabel("y0 - x_mean (MW)")
    ax.set_title(f"Forecast error when using x_mean (raw data)\n({date_range_string})")
    plt.tight_layout()
    plt.savefig(save_path + f"000_y0_minus_xmean_vs_time.png")

    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    ax.plot(y0_minus_xmean.diff())
    y0_minus_xmean.diff().rolling("30D").mean().plot(ax=ax, label="30D roll mean")
    ax.legend()
    ax.set_ylabel("(y0 - x_mean).diff() (MW)")
    ax.set_title(f"Forecast error difference when using x_mean (raw data)\n({date_range_string})")
    plt.tight_layout()
    plt.savefig(save_path + f"000_y0_minus_xmean_diff_vs_time.png")

    for feature_id in features_mw.columns:
        plt.close("all")
        fig, ax = plt.subplots(1, 1)
        ax.scatter(targets_mw["y0"], features_mw[feature_id], s=2, alpha=0.4)
        ax.set_xlabel(f"Power realization y0 (MW)")
        ax.set_ylabel(f"Power forecast difference {feature_id} (MW)")
        ax.set_title(f"Power forecast vs. actual realization (raw data)\n({date_range_string})")
        plt.tight_layout()
        plt.savefig(save_path + f"000_{feature_id}_vs_y0.png")

        plt.close("all")
        fig, ax = plt.subplots(1, 1)
        ax.scatter(targets_mw["y0"].diff(), features_mw[feature_id].diff(), s=2, alpha=0.4)
        ax.set_xlabel(f"Power realization difference y0.diff() (MW)")
        ax.set_ylabel(f"Power forecast difference {feature_id}.diff() (MW)")
        ax.set_title(f"Power forecast diff. vs. actual realization diff. (raw data)\n({date_range_string})")
        plt.tight_layout()
        plt.savefig(save_path + f"000_{feature_id}_diff_vs_y0_diff.png")

    for feature_id in features_mw.columns:
        y0_minus_x = (targets_mw["y0"] - features_mw[feature_id])
        plt.close("all")
        fig, ax = plt.subplots(1, 1)
        y0_minus_x.plot(ax=ax)
        y0_minus_x.rolling("30D").mean().plot(ax=ax)
        ax.set_xlabel(f"")
        ax.set_ylabel(f"y0 - {feature_id} (MW)")
        ax.set_title(f"Vendor {feature_id} forecast error (raw data)")
        plt.tight_layout()
        plt.savefig(save_path + f"000_y0_minus_{feature_id}_vs_time.png")

        plt.close("all")
        fig, ax = plt.subplots(1, 1)
        y0_minus_x.diff().plot(ax=ax)
        y0_minus_x.diff().rolling("30D").mean().plot(ax=ax)
        ax.set_ylabel(f"(y0 - {feature_id}).diff() (MW)")
        ax.set_title(f"Vendor {feature_id} forecast error diff (raw data)")
        plt.tight_layout()
        plt.savefig(save_path + f"000_y0_minus_{feature_id}_diff_vs_time.png")

# -------------------------------------------------------------------------------------------------------------------- #
# Energy production
total_energy_production = round((targets_mw.sum() / 2) * 1e-3, 2)[0]  # GWh

weekly_production = (targets_mw * 0.5).resample("7D").sum()
average_weekly_production = round((targets_mw * 0.5).resample("7D").sum().mean()[0], 2)

weekly_prod_df = targets_mw.copy()
weekly_prod_df["week"] = (targets_mw * 0.5).index.week
production_per_week = weekly_prod_df.groupby("week")["y0"].sum()

plt.close("all")
fig, ax = plt.subplots(1, 1)
production_per_week.plot.bar(ax=ax, label='_nolegend_')
ax.axhline(average_weekly_production, color='k', label=f"Average weekly production ({average_weekly_production}  MWh)")
ax.legend()
ax.set_xlabel("Week number")
ax.set_ylabel(f"Energy production (MWh)")
ax.set_title("Energy production per week")
plt.tight_layout()
plt.savefig(data_path + "energy_production_per_week_bars.png", dpi=120)

plt.close("all")
fig, ax = plt.subplots(1, 1)
targets_mw.plot(ax=ax, label="_nolegend_")
targets_mw["y0"].rolling("30D").mean().plot(ax=ax, label="30D rolling mean")
targets_mw["y0"].rolling("30D").median().plot(ax=ax, label="30D rolling median")
ax.legend()
ax.set_xlabel("")
ax.set_ylabel("Power (MW)")
ax.set_title(f"Actual power production (total energy production {total_energy_production} GWh)")
plt.tight_layout()
plt.savefig(data_path + "power_production_actual.png", dpi=120)

# -------------------------------------------------------------------------------------------------------------------- #
# Identify data spikes

spike_threshold = [2, 1.5, 1]  # Manually determined
for vendor_idx, vendor_id in enumerate(features_mw.columns):

    rolling_quantile = features_mw[vendor_id].diff().abs().rolling("1D").quantile(0.98)

    spikes = (((rolling_quantile > spike_threshold[vendor_idx]).astype(int) -
               (rolling_quantile > spike_threshold[vendor_idx]).astype(int).shift(1)) == 1).astype(int).shift(-1)

    if plotting:
        plt.close("all")
        fig, ax = plt.subplots(1, 1)
        features_mw[vendor_id].plot(ax=ax, color=f"C{vendor_idx}")
        rolling_quantile.plot(ax=ax, color='k', label="1D rolling 98-quantile")
        ax.set_ylabel(f"Power (MW)")
        ax.set_title(f"Power forecast of vendor {vendor_id}")
        plt.tight_layout()
        plt.savefig(save_path + f"000_power_forecast_spikes_vendor{vendor_id}.png")

        plt.close("all")
        fig, ax = plt.subplots(1, 1)
        (spikes*10).plot(ax=ax, color='red', label="Spike to remove", alpha=0.4)
        features_mw[vendor_id].plot(ax=ax, color=f"C{vendor_idx}")
        ax.set_ylabel(f"Power (MW)")
        ax.set_title(f"Power forecast of vendor {vendor_id}")
        plt.tight_layout()
        plt.savefig(save_path + f"000_power_forecast_spikes2remove_vendor{vendor_id}.png")

# -------------------------------------------------------------------------------------------------------------------- #
# Clean data - step 2: Remove unusual spikes and handle nan values

# Remove spikes
features_diff_rolling = features_mw.diff().abs().rolling("1D").quantile(0.98)

spikes_mask = (((features_diff_rolling > spike_threshold).astype(int) -
                (features_diff_rolling > spike_threshold).astype(int).shift(1)) == 1).shift(-1)
spikes_mask.fillna(False, inplace=True)

features_no_spikes = features_mw[~spikes_mask]

# Everything removed?
plt.close("all")
fig, ax = plt.subplots(1, 1)
features_no_spikes.plot(ax=ax)

# One spike remaining - remove that one too
features_no_spikes.loc[pd.Timestamp("2019-05-09 19:30:00"), "x2"] = np.nan

n_spikes_removed = features_no_spikes.isna().astype(int).sum() - features_mw.isna().astype(int).sum()

# Set month where vendor 1 data is corrupted to nan
features_new = features_no_spikes.copy()
features_new.loc["2019-05", "x1"] = np.nan

# Remove time stamps where all predictor variables are nan as we cannot infer value from other predictor variables for
# these time periods
features_new.dropna(how="all", inplace=True)

# Rows where we only have one predictor variable
one_predictor_mask = (features_new.isna().astype(int).sum(axis=1) == 2)
print("Number of rows with one predictor variable: %d" % one_predictor_mask.astype(int).sum())

# Since it is only three rows, drop them for now
features_new = features_new[~one_predictor_mask]
targets_new = targets_mw.loc[features_new.index, :]

vendor_combinations = [(i, j) for i_idx, i in enumerate(feature_names) for j_idx, j in enumerate(feature_names) if (i_idx < j_idx)]
for comb in vendor_combinations:
    vendor_a, vendor_b = comb

    if plotting:
        plt.close("all")
        fig, ax = plt.subplots(1, 1)
        ax.scatter(features_new[vendor_a], features_new[vendor_b], s=2, alpha=0.4)
        ax.set_xlabel(f"Power forecast {vendor_a} (MW)")
        ax.set_ylabel(f"Power forecast {vendor_b} (MW)")
        ax.set_title(f"Forecast correlation after cleaning step 2")
        plt.tight_layout()
        plt.savefig(save_path + f"010_{vendor_a}_vs_{vendor_b}.png")

# Plot histograms of all variables
if plotting:
    _df = pd.concat([features_new, targets_new], axis=1)

    for var in _df.columns:
        plt.close("all")
        fig, ax = plt.subplots(1, 1)
        _df[var].hist(ax=ax, bins=100)
        ax.set_xlabel(f"{var} (MW)")
        ax.set_ylabel(f"# occurrences")
        ax.set_title(f"Histogram of {var} after cleaning step 2")
        plt.tight_layout()
        plt.savefig(save_path + f"011_{var}_hist.png")

    error_mean = _df["y0"] - _df[feature_names].mean(axis=1)
    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    error_mean.hist(ax=ax, bins=100)
    ax.set_xlabel(f"Error of x_mean (MW)")
    ax.set_ylabel(f"# occurrences")
    ax.set_title(f"Histogram of x_mean error after cleaning step 2")
    plt.tight_layout()
    plt.savefig(save_path + f"011_x_mean_error_hist.png")

    for vendor_id in feature_names:
        error_vendor = _df["y0"] - _df[vendor_id]
        plt.close("all")
        fig, ax = plt.subplots(1, 1)
        error_vendor.hist(ax=ax, bins=100)
        ax.set_xlabel(f"Error of {vendor_id} (MW)")
        ax.set_ylabel(f"# occurrences")
        ax.set_title(f"Histogram of {vendor_id} error after cleaning step 2")
        plt.tight_layout()
        plt.savefig(save_path + f"012_{vendor_id}_error_hist.png")

# -------------------------------------------------------------------------------------------------------------------- #
# Clean data - step 3: Predict nan values appearing in vendor x1 column based on forecast of other two vendors
# (1) Look at how prediction works when training with all data (except for last day) and testing with last day
#     (a) with original features/targets
#     (b) with difference series of original features/targets
# (2) Benchmark with (i) mean and (ii) best vendor
# (3) Choose model based on results
# -------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
# (1) Handle remaining in nans in features and targets

vendor_id = "x1"
other_vendors = set(feature_names).difference(set([vendor_id]))

x_df = features_new[other_vendors].copy()
y_df = features_new[[vendor_id]].copy()

# All potential training data
training_mask = y_df[~y_df[vendor_id].isna()].index
x_train_df = x_df.loc[training_mask, :]
y_train_df = y_df.loc[training_mask, :]

# All potential testing data
test_mask = y_df[y_df[vendor_id].isna()].index
x_test_df = x_df.loc[test_mask, :]

if plotting:
    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    x_train_df.plot(ax=ax)
    (x_train_df.isna().astype(int) * 10).plot(ax=ax, color='k', alpha=0.2, label='_nolegend_')
    ax.set_xlabel("")
    ax.set_ylabel(f"Power forecast (MW)")
    ax.set_title(f"Power forecast after cleaning step 2\nStill nans remaining")

# Some of the predictor variables might still be nan: Fill with mean of two neighboring data points for each series
for col in x_train_df.columns:
    t_nan = x_train_df.loc[x_train_df[col].isna(), col].index
    t_m1 = [x_train_df.loc[:t_nan[i], col].dropna()[-1] for i in range(len(t_nan))]  # last available point
    t_p1 = [x_train_df.loc[t_nan[i]:, col].dropna()[0] for i in range(len(t_nan))]

    fill_nan_values = (np.array(t_m1) + np.array(t_p1)) / 2
    x_train_df.loc[t_nan, col] = fill_nan_values

# -------------------------------------------------------------------------------------------------------------------- #
# Define test and training sets:
# (1a) Prediction with original features/targets for last available day based on preceding data

x_test_a = x_train_df.last("1D").copy()
y_test_a = y_train_df.last("1D").copy()

x_train_a = x_train_df.loc[:y_test_a.index[0] - pd.Timedelta(t_res), :]
y_train_a = y_train_df.loc[:y_test_a.index[0] - pd.Timedelta(t_res), :]

if plotting:
    _title_ = f"Train and test period (1a)"
    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    x_train_a.resample("30T").asfreq(np.nan).plot(ax=ax, color='k', alpha=0.5, legend=False)
    y_train_a.resample("30T").asfreq(np.nan).plot(ax=ax, color='k', alpha=0.5, legend=False)
    x_test_a.plot(ax=ax, color='r', legend=False)
    y_test_a.plot(ax=ax, color='r', legend=False)
    ax.set_title(_title_)
    plt.tight_layout()
    plt.savefig(save_path + f"020_data_cleaning_train_and_test_1a.png")

# -------------------------------------------------------------------------------------------------------------------- #
# Prediction 1a

x_train_vals = x_train_a.values
y_train_vals = y_train_a.values
x_test_vals = x_test_a.values
y_test_vals = y_test_a.values

# Linear regression
my_regression = MyLinearRegression()
my_regression.fit(x_train_vals, y_train_vals)
my_pred_1a = my_regression.predict(x_test_vals)
my_error_1a = error(y_test_vals, my_pred_1a)
my_score_1a = my_regression.score(y_test_vals, my_pred_1a, metric="MSE")
my_mae_score_1a = my_regression.score(y_test_vals, my_pred_1a, metric="MAE")
my_weights_1a = my_regression.weights

x_mean_error_1a, x_mean_score_1a = my_regression.benchmark_mean(x_test_vals, y_test_vals, metric="MSE")
vendor_error_1a, vendor_score_1a = my_regression.benchmark_vendor(x_test_vals, y_test_vals, metric="MSE")
_, x_mean_mae_score_1a = my_regression.benchmark_mean(x_test_vals, y_test_vals, metric="MAE")
_, vendor_mae_score_1a = my_regression.benchmark_vendor(x_test_vals, y_test_vals, metric="MAE")

error_1a_df = pd.DataFrame(data=my_error_1a, index=x_test_a.index, columns=["My error"])
error_1a_df["x_mean error"] = x_mean_error_1a
for _idx, _id in enumerate(other_vendors):
    error_1a_df[f"vendor {_id} error"] = vendor_error_1a[:, _idx]

score_1a_df = pd.Series(data=my_score_1a, index=["My MSE score"])
score_1a_df.loc["x_mean MSE score"] = x_mean_score_1a[0]
for _idx, _id in enumerate(other_vendors):
    score_1a_df.loc[f"vendor {_id} MSE score"] = vendor_score_1a[_idx]

mae_score_1a_df = pd.Series(data=my_mae_score_1a, index=["My MAE score"])
mae_score_1a_df.loc["x_mean MAE score"] = x_mean_mae_score_1a[0]
for _idx, _id in enumerate(other_vendors):
    mae_score_1a_df.loc[f"vendor {_id} MAE score"] = vendor_mae_score_1a[_idx]

y_comparison = pd.DataFrame(data=my_pred_1a, index=x_test_a.index, columns=["My pred"])
y_comparison["x_mean"] = np.mean(x_test_vals, axis=1)
for _idx, _id in enumerate(other_vendors):
    y_comparison[f"{_id}"] = x_test_vals[:, _idx]
y_comparison["y_is"] = y_test_vals

if plotting:
    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    y_comparison.plot(ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Power (MW)")
    ax.set_title("Predict last 24 hours based on all preceding data (1a)")
    plt.tight_layout()
    plt.savefig(save_path + f"020_y_comparison_1a.png", dpi=120)

    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    error_1a_df.plot(ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Error (MW)")
    ax.set_title("Predict last 24 hours based on all preceding data (1a)")
    plt.tight_layout()
    plt.savefig(save_path + f"020_error_comparison_1a.png", dpi=120)

    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    score_1a_df.plot.bar(ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("MSE")
    ax.set_title("Predict last 24 hours based on all preceding data (1a)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(save_path + f"020_score_comparison_1a.png", dpi=120)

    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    mae_score_1a_df.plot.bar(ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("MAE")
    ax.set_title("Predict last 24 hours based on all preceding data (1a)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(save_path + f"020_MAE_score_comparison_1a.png", dpi=120)

# -------------------------------------------------------------------------------------------------------------------- #
# Even though vendor 2 has lowest error, let's take mean to fill nan values of vendor 1
# These data will be used further

if "features_clean_mw.pkl" not in os.listdir(data_path):
    features_clean = pd.concat([x_train_df, x_test_df])
    features_clean.loc[:, vendor_id] = y_train_df
    features_clean.loc[x_test_df.index, vendor_id] = features_clean.loc[x_test_df.index, other_vendors].mean(axis=1)
    features_clean.sort_index(inplace=True)
    features_clean = features_clean[feature_names]

    features_clean_mw = features_clean.copy()
    target_clean_mw = targets_new.loc[features_clean_mw.index, :].copy()

    pickle.dump(features_clean_mw, open(data_path + "features_clean_mw.pkl", "wb"))
    pickle.dump(target_clean_mw, open(data_path + "target_clean_mw.pkl", "wb"))

# -------------------------------------------------------------------------------------------------------------------- #
# These results are not used any further
# Prediciton (1b) - based on differences

vendor_id = "x1"
other_vendors = set(feature_names).difference(set([vendor_id]))

x_diff_df = features_new[other_vendors].diff().iloc[1:].copy()
y_diff_df = features_new[[vendor_id]].diff().iloc[1:].copy()

# All potential training data
training_mask = y_diff_df[~y_diff_df[vendor_id].isna()].index
x_diff_train_df = x_diff_df.loc[training_mask, :]
y_diff_train_df = y_diff_df.loc[training_mask, :]

# All potential testing data
test_mask = y_diff_df[y_diff_df[vendor_id].isna()].index
x_diff_test_df = x_diff_df.loc[test_mask, :]

# Some of the predictor variables might still be nan: Fill with mean of two neighboring data points for each series
for col in x_diff_train_df.columns:
    t_nan = x_diff_train_df.loc[x_diff_train_df[col].isna(), col].index
    t_m1 = [x_diff_train_df.loc[:t_nan[i], col].dropna()[-1] for i in range(len(t_nan))]  # last available point
    t_p1 = [x_diff_train_df.loc[t_nan[i]:, col].dropna()[0] for i in range(len(t_nan))]

    fill_nan_values = (np.array(t_m1) + np.array(t_p1)) / 2
    x_diff_train_df.loc[t_nan, col] = fill_nan_values

# -------------------------------------------------------------------------------------------------------------------- #
# Define test and training sets:
# (1b) Prediction with diff of features/targets for last available day based on preceding data

x_diff_test_b = x_diff_train_df.last("1D").copy()
y_diff_test_b = y_diff_train_df.last("1D").copy()

x_diff_train_b = x_diff_train_df.loc[:y_diff_test_b.index[0] - pd.Timedelta(t_res), :]
y_diff_train_b = y_diff_train_df.loc[:y_diff_test_b.index[0] - pd.Timedelta(t_res), :]

if plotting:
    _title_ = f"Train and test period (1b)"
    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    x_diff_train_b.resample("30T").asfreq(np.nan).plot(ax=ax, color='k', alpha=0.5, legend=False)
    y_diff_train_b.resample("30T").asfreq(np.nan).plot(ax=ax, color='k', alpha=0.5, legend=False)
    x_diff_test_b.plot(ax=ax, color='r', legend=False)
    y_diff_test_b.plot(ax=ax, color='r', legend=False)
    ax.set_title(_title_)
    plt.tight_layout()
    plt.savefig(save_path + f"020_data_cleaning_train_and_test_1b.png")

# -------------------------------------------------------------------------------------------------------------------- #
# Prediction 1b

x_diff_train_vals = x_diff_train_b.values
y_diff_train_vals = y_diff_train_b.values
x_diff_test_vals = x_diff_test_b.values
y_diff_test_vals = y_diff_test_b.values

# Linear regression
my_regression = MyLinearRegression()
my_regression.fit(x_diff_train_vals, y_diff_train_vals)
my_pred_1b = my_regression.predict(x_diff_test_vals)

# From last know value compute prediction in MW
y_0 = features_new.loc[y_diff_test_b.index[0] - pd.Timedelta("30T"), vendor_id]
my_pred_1b_mw = []
for i in my_pred_1b:
    new_val = i[0] + y_0
    my_pred_1b_mw.append(new_val)
    y_0 =  new_val
my_pred_1b_mw = np.array([my_pred_1b_mw]).reshape(-1, 1)

my_error_1b = error(y_test_vals, my_pred_1b_mw)
my_score_1b = my_regression.score(y_test_vals, my_pred_1b_mw, metric="MSE")
my_mae_score_1b = my_regression.score(y_test_vals, my_pred_1b_mw, metric="MAE")
my_weights_1b = my_regression.weights

# -------------------------------------------------------------------------------------------------------------------- #
# Plot
y_comparison.rename({"My pred": "My pred (1a)"}, axis=1, inplace=True)
y_comparison["My pred (1b)"] = my_pred_1b_mw

error_df = error_1a_df.copy()
error_df.rename({"My error": "My error (1a)"}, axis=1, inplace=True)
error_df["My error (1b)"] = my_error_1b

score_df = score_1a_df.copy()
score_df.rename({"My MSE score": "My MSE score (1a)"}, axis=1, inplace=True)
score_df["My MSE score (1b)"] = my_score_1b[0]

mae_score_df = mae_score_1a_df.copy()
mae_score_df.rename({"My MAE score": "My MAE score (1a)"}, axis=0, inplace=True)
score_df["My MAE score (1b)"] = my_mae_score_1b[0]

if plotting:
    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    y_comparison.plot(ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Power (MW)")
    ax.set_title("Predict last 24 hours based on all preceding data")
    plt.tight_layout()
    plt.savefig(save_path + f"030_y_comparison_both_models.png", dpi=120)

    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    error_df.plot(ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Error (MW)")
    ax.set_title("Predict last 24 hours based on all preceding data")
    plt.tight_layout()
    plt.savefig(save_path + f"030_error_comparison_both_models.png", dpi=120)

    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    score_df.plot.bar(ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("MSE")
    ax.set_title("Predict last 24 hours based on all preceding data")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(save_path + f"030_score_comparison_both_models.png", dpi=120)

    plt.close("all")
    fig, ax = plt.subplots(1, 1)
    mae_score_df.plot.bar(ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("MAE")
    ax.set_title("Predict last 24 hours based on all preceding data")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(save_path + f"030_MAE_score_comparison_both_models.png", dpi=120)





