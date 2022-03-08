from datetime import datetime
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from typing import Tuple
from datasets.setting import DATASETS_PATH


DATA_DIR = os.path.join(DATASETS_PATH, 'caiso', 'caiso_20130101_20210630.csv')

@dataclass()
class CaisoMeta:
    horizon = 24
    seasonal_pattern = 'Hourly'
    frequency = 24

@dataclass()
class CaisoDataset:
    ids: np.ndarray
    values: np.ndarray
    dates: np.ndarray

    @staticmethod
    def load() -> 'CaisoDataset':
        """
        Get CaisoDataset dataset.
        """
        data = pd.read_csv(DATA_DIR)
        data['Date'] = data['Date'].astype('datetime64')
        names = ['PGE','SCE','SDGE','VEA','CA ISO','PACE','PACW','NEVP','AZPS','PSEI']
        ids = np.arange(len(names))
        df_all = pd.DataFrame(pd.date_range('20130101','20210630',freq='H')[:-1], columns=['Date'])
        for name in names:
            current_df = data[data['zone'] == name].drop_duplicates(subset='Date', keep='last').rename(columns={'load':name}).drop(columns=['zone'])
            df_all = df_all.merge(current_df, on='Date', how='outer')
        # set index
        df_all = df_all.set_index('Date')
        values = df_all.fillna(0).values.T
        dates = np.array([str(x)[:13] for x in df_all.index.tolist()])
        return CaisoDataset(ids=ids, values=values, dates=dates)

    def split_by_date(self, cut_date: str, include_cut_date: bool = True) -> Tuple['CaisoDataset', 'CaisoDataset']:
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
        return CaisoDataset(ids=self.ids,
                                  values=self.values[:, left_indices],
                                  dates=self.dates[left_indices]), \
               CaisoDataset(ids=self.ids,
                                  values=self.values[:, right_indices],
                                  dates=self.dates[right_indices])

    def split(self, cut_point: int) -> Tuple['CaisoDataset', 'CaisoDataset']:
        """
        Split dataset by cut point.

        :param cut_point: Cut index.
        :return: Two parts of dataset: left contains all points before the cut point
        and the right part contains all datpoints on and after the cut point.
        """
        return CaisoDataset(ids=self.ids,
                                  values=self.values[:, :cut_point],
                                  dates=self.dates[:cut_point]), \
               CaisoDataset(ids=self.ids,
                                  values=self.values[:, cut_point:],
                                  dates=self.dates[cut_point:])

    def time_points(self):
        return self.dates.shape[0]