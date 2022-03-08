import pandas as pd
import numpy as np
from evaluation.metrics import *
from evaluation.eval_utils import median_ensemble
from datasets.traffic import TrafficDataset
from datasets.electricity import ElectricityDataset
from datasets.caiso import CaisoDataset
from datasets.nordpool import ProductionDataset


summary_filter = 'seed:*,lookback:*'
test_lengths = 7 * 24


# evaluation for electricity
electricity_splits = {'deepar': '2014-08-31 23', 'last': '2014-12-25 00'}

data_name = 'electricity'
split_name = 'deepar'
training_set, test_set = ElectricityDataset.load().split_by_date(electricity_splits[split_name])
forecast = median_ensemble(f'output/{data_name}', forecast_file=f'forecast_{split_name}.csv', summary_filter=summary_filter)
target = test_set.values[:,:test_lengths]
print(nd(forecast, target))

data_name = 'electricity'
split_name = 'last'
training_set, test_set = ElectricityDataset.load().split_by_date(electricity_splits[split_name])
forecast = median_ensemble(f'output/{data_name}', forecast_file=f'forecast_{split_name}.csv', summary_filter=summary_filter)
target = test_set.values[:,:test_lengths]
print(nd(forecast, target))


# evaluation for traffic
traffic_splits = {'deepar': '2008-06-14 23', 'last': '2009-03-24 00'}

data_name = 'traffic'
split_name = 'deepar'
training_set, test_set = TrafficDataset.load().split_by_date(traffic_splits[split_name])
forecast = median_ensemble(f'output/{data_name}', forecast_file=f'forecast_{split_name}.csv', summary_filter=summary_filter)/100000
target = test_set.values[:,:test_lengths]
print(nd(forecast, target))

data_name = 'traffic'
split_name = 'last'
training_set, test_set = TrafficDataset.load().split_by_date(traffic_splits[split_name])
forecast = median_ensemble(f'output/{data_name}', forecast_file=f'forecast_{split_name}.csv', summary_filter=summary_filter)/100000
target = test_set.values[:,:test_lengths]
print(nd(forecast, target))


# evaluation for caiso
caiso_splits = {"last18months":"2020-01-01 00", "last15months":"2020-04-01 00",
                "last12months":"2020-07-01 00", "last9months":"2020-10-01 00"}

for split_name in ["last18months","last15months","last12months","last9months"]:
    data_name = 'caiso'
    training_set, test_set = CaisoDataset.load().split_by_date(caiso_splits[split_name], False)
    forecast = median_ensemble(f'output/{data_name}', forecast_file=f'forecast_{split_name}.csv', summary_filter=summary_filter)
    target = test_set.values[:,:test_lengths]
    print(nd(forecast, target), nrmse(forecast, target))


# evaluation for np
np_splits = {"last12months":"2020-01-01 00","last9months":"2020-04-01 00",
             "last6months":"2020-07-01 00", "last3months":"2020-10-01 00"}

for split_name in ["last12months","last9months","last6months","last3months"]:
    data_name = 'product'
    training_set, test_set = ProductionDataset.load().split_by_date(np_splits[split_name], False)
    forecast = median_ensemble(f'output/{data_name}', forecast_file=f'forecast_{split_name}.csv', summary_filter=summary_filter)
    target = test_set.values[:,:test_lengths]
    print(nd(forecast, target), nrmse(forecast, target))