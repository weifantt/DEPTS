# adopt from https://github.com/ElementAI/N-BEATS
import os
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

def median_ensemble(experiment_path: str,
                    summary_filter: str = '**',
                    forecast_file: str = 'forecast.csv',
                    group_by: str = 'id'):
    """
    Build a median ensemble from files found in the experiment path.

    :param experiment_path: Experiment path.
    :param summary_filter: Filter which experiment instances should be included in ensemble.
    :param forecast_file: Name of the file with results.
    :param group_by: Grouping key.
    :return: Pandas dataframe with median forecasts.
    """
    return pd.concat([pd.read_csv(file)
                      for file in
                      tqdm(glob(os.path.join(experiment_path, summary_filter, forecast_file)))], sort=False) \
        .set_index(group_by).groupby(level=group_by, sort=False).median().values

def group_values(values: np.ndarray, groups: np.ndarray, group_name: str) -> np.ndarray:
    """
    Filter values array by group indices and clean it from NaNs.

    :param values: Values to filter.
    :param groups: Timeseries groups.
    :param group_name: Group name to filter by.
    :return: Filtered and cleaned timeseries.
    """
    return np.array([v[~np.isnan(v)] for v in values[groups == group_name]])


def test_path(experiment_path: str,
                    summary_filter: str = '**',
                    forecast_file: str = 'forecast.csv',
                    group_by: str = 'id'):
    for file in tqdm(glob(os.path.join(experiment_path, summary_filter, forecast_file))):
        print(file)


import matplotlib.pyplot as plt
def plot_one_series(local_list):
    plt.figure(1)
    plt.plot(range(len(local_list)), local_list, color='b')
    plt.show()

def plot_two_series(local_list, mix_list):
    plt.figure(1)
    plt.plot(range(len(local_list)), local_list, color='b')
    plt.plot(range(len(mix_list)), mix_list, color='r')
    plt.show()