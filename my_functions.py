import numpy as np
import os
import pandas as pd
import time


def adjust_data_type(df, data_type="float"):
    """Adjust data type of input data frame to required data type.

    Parameters
    ----------
    df : pandas dataframe
         The data used to compute the mean and standard deviation
         used for later scaling along the features axis.

    """

    if not np.all(df.dtypes.unique() == data_type):
        raise TypeError(f"Input data adjusted to {data_type}.")

    df_new = df.astype(data_type)
    return df_new

def createdir(path):
    """Checks existance of a directory and creates it, if it does not exit.
    :param path: str, path to check and create
    :return: str, path created or not.
    """
    if path[-1] != '/':
        path = path + '/'
    if not os.path.exists(path):
        new_path = path
        os.makedirs(new_path)
    else:
        new_path = path
    return new_path

def timer(func_to_time):
    """This is a decorator for timing execution of any input function. To use it, the decorator must be written
    together with the definition of the function like the example below:

    @timer
    def my_function(*args,**kwargs):
        do something...
        return whatever

    :param func_to_time: function (object)
    :return: returns the same function, decorated with the timer function
    """

    def decorator(*args, **kwargs):
        print(f"Running {func_to_time.__name__}():", end=" ")
        t0 = time.time()
        results = func_to_time(*args, **kwargs)
        run_time_secs = time.time() - t0
        print(f"\t{get_time_in_nice_format(run_time_secs)}")
        # display_time_in_nice_format(run_time_secs, message='{}()'.format(func_to_time.__name__), jumpline=False)
        return results

    return decorator

def get_time_in_nice_format(time_in_secs):
    if time_in_secs < 60:
        display_time = '{} sec'.format(round(time_in_secs, 2))
    elif (time_in_secs >= 60) and (time_in_secs < 3600):
        mins = int(time_in_secs / 60)
        secs = int(time_in_secs % 60)
        display_time = '{} min {} sec'.format(mins, secs)
    elif (time_in_secs >= 3600) and (time_in_secs < 24 * 3600):
        hours = int(time_in_secs / 3600)
        mins = int((time_in_secs % 3600) / 60)
        display_time = '{} hour {} min'.format(hours, mins)
    else:
        days = int(time_in_secs / (24 * 3600))
        hours = int((time_in_secs % (24 * 3600)) / 3660)
        mins = int((time_in_secs - 24 * 3600 * days - 3600 * hours) / 60)
        display_time = '{} day {} hour {} mins'.format(days, hours, mins)
    return display_time

def check_for_any_nans(x):
    """ Check for any nans in numpy array """
    return np.isnan(x).any()