from datetime import datetime
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from typing import Tuple
from datasets.setting import DATASETS_PATH


@dataclass()
class ProductionMeta:
    horizon = 24
    seasonal_pattern = 'Hourly'
    frequency = 24


@dataclass()
class ProductionDataset:
    ids: np.ndarray
    values: np.ndarray
    dates: np.ndarray

    @staticmethod
    def load() -> 'ProductionDataset':
        """
        Get NordPoolProductionDataset dataset.
        """
        DATA_PATH = os.path.join(DATASETS_PATH, 'nordpool', 'production.csv')
        data = pd.read_csv(DATA_PATH, parse_dates=['Time'])
        data = data.set_index('Time')
        ids = np.arange(data.shape[1])
        values = data.fillna(0).values.T
        dates = np.array([str(x)[:13] for x in data.index.tolist()])

        return ProductionDataset(ids=ids, values=values, dates=dates)

    def split_by_date(self, cut_date: str, include_cut_date: bool = True) -> Tuple['ProductionDataset', 'ProductionDataset']:
        """
        Split dataset by date.

        :param cut_date: Cut date in "%Y-%m-%d %H" format
        :param include_cut_date: Include cut_date in the split if true, not otherwise.
        :return: Two parts of dataset: the left part contains all points before the cut point
        and the right part contains all datpoints on and after the cut point.
        """
        date = datetime.strptime(cut_date, '%Y-%m-%d %H')
        left_indices = []
        right_indices = []
        for i, p in enumerate(self.dates):
            record_date = datetime.strptime(p, '%Y-%m-%d %H')
            if record_date < date or (include_cut_date and record_date == date):
                left_indices.append(i)
            else:
                right_indices.append(i)
        return ProductionDataset(ids=self.ids,
                                  values=self.values[:, left_indices],
                                  dates=self.dates[left_indices]), \
               ProductionDataset(ids=self.ids,
                                  values=self.values[:, right_indices],
                                  dates=self.dates[right_indices])

    def split(self, cut_point: int) -> Tuple['ProductionDataset', 'ProductionDataset']:
        """
        Split dataset by cut point.

        :param cut_point: Cut index.
        :return: Two parts of dataset: left contains all points before the cut point
        and the right part contains all datpoints on and after the cut point.
        """
        return ProductionDataset(ids=self.ids,
                                  values=self.values[:, :cut_point],
                                  dates=self.dates[:cut_point]), \
               ProductionDataset(ids=self.ids,
                                  values=self.values[:, cut_point:],
                                  dates=self.dates[cut_point:])

    def time_points(self):
        return self.dates.shape[0]

