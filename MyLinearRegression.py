import numpy as np
import pandas as pd
from my_functions import timer

def error(y_is, y_pred):
    """
    :param  y_is: numpy array of shape (n_observations, 1)
    :param  y_pred: numpy array of shape (n_observations, 1)
    :return: error
    """
    return (y_is - y_pred)

def mean_absolute_error(y_is, y_pred):
    return np.sum(np.abs(error(y_is, y_pred)), axis=0) / len(y_is)

def mean_square_error(y_is, y_pred):
    return (np.sum(error(y_is, y_pred), axis=0) ** 2) / len(y_is)

def root_mean_square_error(y_is, y_pred):
    return np.sqrt(mean_square_error(y_is, y_pred))

def sliding_windows(x_df, window_size="30D", test_size="24H", t_res="30T"):
    """ Create rolling/sliding blocks: trainings set of length window_size, testing set of length test_size.
    :param  x_df: numpy array of shape (n_observations, 1)
    :return: list of lists with time indices for training sets and test sets
    """

    train_windows = []
    test_windows = []
    t_start = x_df.index.min()
    t_end = t_start + pd.Timedelta(window_size)
    while (t_end - pd.Timedelta(t_res) <= x_df.index.max() - pd.Timedelta(test_size)):
        t_training = x_df.loc[t_start:t_end - pd.Timedelta(t_res), :].index
        t_testing = x_df.loc[t_end:t_end + pd.Timedelta(test_size) - pd.Timedelta(t_res), :].index

        train_windows.append(t_training)
        test_windows.append(t_testing)

        t_start = t_start + pd.Timedelta('1D')
        t_end = t_start + pd.Timedelta(window_size)

    return train_windows, test_windows


class MyLinearRegression:

    def __init__(self):
        self.error = []
        self.weights = []
        self.n_features = 0
        self.n_observations = 0
        self.n_predictions = 0
        self.x_test = []

    @timer
    def fit(self, x, y):

        self.data_integrity(x, y)

        self.n_observations = x.shape[0]
        self.n_features = x.shape[1]

        if len(x.shape) == 1:
            x.reshape(-1, 1)

        # Add ones for intercept vector
        x = self.add_ones(x)

        self.weights = np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)

    def predict(self, x_test):
        """ Predict target variable based on new observations x_new """
        self.x_test = x_test

        w_0 = self.weights[0]
        w_i = self.weights[1:]

        self.n_predictions = x_test.shape[0]

        w_i = w_i.reshape(1, self.n_features)

        if self.n_features > 1:
            y_pred = w_0 + np.sum(x_test * w_i, axis=1)
        else:
            y_pred = w_0 + x_test * w_i

        return y_pred.reshape(-1, 1)

    def add_ones(self, x):
        """ Add a vector of ones to matrix of predictor variables (intercept) """
        self.n_observations = x.shape[0]
        ones = np.ones(x.shape[0]).reshape(-1, 1)

        return np.concatenate((ones, x), axis=1)

    def data_integrity(self, x, y):
        if np.isnan(x).any() or np.isnan(y).any():
            raise ValueError(
                f"{self.__class__.__name__}: "
                f"Features or target variables contain nan values. Linear regression won't work."
            )

    def score(self, y_is, y_pred, metric="MSE"):
        if metric == "MSE":
            return mean_square_error(y_is, y_pred)
        elif metric == "RMSE":
            return root_mean_square_error(y_is, y_pred)
        elif metric == "MAE":
            return mean_absolute_error(y_is, y_pred)

    def benchmark_mean(self, x_vendors, y_is, metric="MSE"):
        """ Benchmark test set with mean of vendor forecast """
        x_mean = np.mean(x_vendors, axis=1).reshape(-1, 1)
        x_mean_error = error(y_is, x_mean)
        x_mean_score = self.score(y_is, x_mean, metric=metric)

        return x_mean_error, x_mean_score

    def benchmark_vendor(self, x_vendors, y_is, metric="MSE"):
        """ Benchmark with each vendor forecast individually """
        vendor_error = error(y_is, x_vendors)
        vendor_score = self.score(y_is, x_vendors, metric=metric)

        return vendor_error, vendor_score

    def skill_score(self, my_score, benchmark_score):
        """ Skill score to determine relative quality of a method vs. a benchmark """
        return (1 - my_score/benchmark_score)

    def rolling_regression(self):
        pass

    def cross_validation(self):
        pass


