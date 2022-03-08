import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import argparse
import random
from pathlib import Path
import datetime
from utils import TimeseriesSampler, to_tensor, warm_PM_parameters_perK
from utils import default_device, mape_loss, mase_loss, smape_2_loss
from models import PeriodicityModule, depts_expansion_general
from datasets.electricity import ElectricityDataset, ElectricityMeta
from datasets.traffic import TrafficDataset, TrafficMeta
from datasets.caiso import CaisoDataset, CaisoMeta
from datasets.nordpool import ProductionDataset, ProductionMeta

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def __loss_fn(loss_name: str):
    def loss(x, freq, forecast, target, target_mask):
        if loss_name == 'MAPE':
            return mape_loss(forecast, target, target_mask)
        elif loss_name == 'MASE':
            return mase_loss(x, freq, forecast, target, target_mask)
        elif loss_name == 'SMAPE':
            return smape_2_loss(forecast, target, target_mask)
        else:
            raise Exception(f'Unknown loss function: {loss_name}')
    return loss

parser = argparse.ArgumentParser(description='Parameters')
# choosing data and split
parser.add_argument('--data', type=str, default='')
parser.add_argument('--split', type=str, default='')
# choosing general setting
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lookback', type=int, default=2)  # 2, 3, 4, 5, 6, 7
parser.add_argument('--loss', type=str, default='SMAPE') # SMAPE
parser.add_argument('--output', type=str, default='') # output file dirs
# choose parameters
parser.add_argument('--J', type=int, default=8) # J valid periods: 4, 8, 16, 32
parser.add_argument('--maxK', type=int, default=128) # in top-K candidate periods to select J valid periods
parser.add_argument('--period_layers', type=int, default=1) # num of NN layers in periodic blocks
args = parser.parse_args()


DATA_CONFIG = {
    "electricity":{"data":ElectricityDataset,"meta":ElectricityMeta, "fftwarmlen": 24*10,
                   'deepar':"2014-08-31 23","last":"2014-12-25 00",
                   "lr":0.001, "history_size":10,"test_windows":7, "iters":72000, "dump_iters":[72000]},
    "traffic":{"data":TrafficDataset,"meta":TrafficMeta, "fftwarmlen": 24*30*2,
               'deepar':"2008-06-14 23", "last":"2009-03-24 00",
               "lr":0.001,"history_size": 10,"test_windows":7, "iters":24000, "dump_iters":[12000]},
    "caiso":{"data":CaisoDataset, "meta":CaisoMeta, "fftwarmlen": 24*30*24,
             "last18months":"2020-01-01 00", "last15months":"2020-04-01 00",
             "last12months":"2020-07-01 00", "last9months":"2020-10-01 00",
             "history_size":30*24,"test_windows":7,"lr":0.001,"iters":4000, "dump_iters":[4000]},
    "product":{"data":ProductionDataset, "meta":ProductionMeta, "fftwarmlen": 24*30*24,
               "last12months":"2020-01-01 00","last9months":"2020-04-01 00",
               "last6months":"2020-07-01 00", "last3months":"2020-10-01 00",
               "history_size":30*24,"test_windows":7,"lr":1e-6,"iters":12000, "dump_iters":[12000]},
}

# Set parameters
setup_seed(args.seed)
data_name = args.data
split_name = args.split
lookback = args.lookback
loss_name = args.loss
OUTPUT_DIR = args.output
maxK, J = args.maxK, args.J
period_layers = args.period_layers
EM_learning_rate = DATA_CONFIG[data_name]['lr']
sum_iterations = DATA_CONFIG[data_name]['iters']
dump_iters = DATA_CONFIG[data_name]['dump_iters']
history_size = DATA_CONFIG[data_name]['history_size']
fftwarmlen = DATA_CONFIG[data_name]['fftwarmlen']
PM_learning_rate = 5e-7
test_windows = 7
generic_layer_size, generic_layers, generic_stacks = 512, 4, 30
batch_size = 1024

dataset = DATA_CONFIG[data_name]['data'].load()
split_date = DATA_CONFIG[data_name][split_name]
horizon = DATA_CONFIG[data_name]['meta'].horizon
frequency = DATA_CONFIG[data_name]['meta'].frequency
input_size = lookback * horizon


if data_name == 'electricity' or data_name == 'traffic':
    training_set, test_set = dataset.split_by_date(split_date)
else:
    training_set, test_set = dataset.split_by_date(split_date, False)

training_values = training_set.values.astype('float64')
test_values = test_set.values.astype('float64')

print("dataset reading, done!")

# fft transform is sensitive to data scale, we do a simple scaling for traffic dataset
# during test, the predicted values are recovered to raw scale
if data_name == 'traffic':
    training_values, test_values = map(lambda x:x*100000,(training_values, test_values))

training_sampler = TimeseriesSampler(timeseries=training_values, insample_size=input_size,
                outsample_size=horizon, window_sampling_limit=history_size * horizon)
iter_training_sampler = iter(training_sampler)



# parameters init for Periodicity Module
fft_warm_point = training_values.shape[1] - fftwarmlen
all_a, all_p, all_f, all_mean, all_Kmask = warm_PM_parameters_perK(training_values, fft_warm_point, maxK, J)
PM = PeriodicityModule(all_a=all_a, all_p=all_p, all_f=all_f, all_mean=all_mean).to(default_device())
all_Kmask = to_tensor(all_Kmask)
PM_optimizer = optim.Adam(PM.parameters(), lr=PM_learning_rate)
print("PM init, done!")

# parameters init for Expansion Module
EM = depts_expansion_general(input_size=input_size, output_size=horizon, layer_size=generic_layer_size, stacks=generic_stacks,
            local_layers=generic_layers, period_layers=period_layers, num_series=training_values.shape[0]).to(default_device())
EM_optimizer = optim.Adam(EM.parameters(), lr=EM_learning_rate)
print("EM init, done!")


training_loss_fn = __loss_fn(loss_name)
lr_decay_step = sum_iterations // 3
if lr_decay_step == 0:
    lr_decay_step = 1

# begin training
training_loss_log = []
print("start", split_name)
for i in range(sum_iterations + 1):

    if i % 1000 == 0:
        print(i, datetime.datetime.now())

    PM.train()
    EM.train()
    x, x_mask, y, y_mask, x_timestamp, y_timestamp, ids = map(to_tensor, next(iter_training_sampler))

    EM_optimizer.zero_grad()
    PM_optimizer.zero_grad()

    batchKmask = all_Kmask.index_select(0, ids.long().view(-1))
    x_z, y_z = PM(x_timestamp, ids.long(), batchKmask), PM(y_timestamp, ids.long(), batchKmask)
    forecast, _, _ = EM(x, x_z, x_mask, y_z, ids)
    training_loss = training_loss_fn(x, ElectricityMeta.frequency, forecast, y, y_mask)
    training_loss_log.append(training_loss.item())

    if np.isnan(float(training_loss)):
        break

    training_loss.backward()

    torch.nn.utils.clip_grad_norm_(EM.parameters(), 1.0)
    EM_optimizer.step()
    for param_group in EM_optimizer.param_groups:
        param_group["lr"] = EM_learning_rate * 0.5 ** (i // lr_decay_step)

    torch.nn.utils.clip_grad_norm_(PM.parameters(), 1.0)
    PM_optimizer.step()
    for param_group in PM_optimizer.param_groups:
        param_group["lr"] = PM_learning_rate * 0.5 ** (i // lr_decay_step)


    if i in dump_iters:
        # Build rolling forecasts
        # We call periodic block output as global_forecasts
        # We call local block output as local_forecasts
        forecasts = []
        global_forecasts = []
        local_forecasts = []
        z_outputs = []
        PM.eval()
        EM.eval()
        with torch.no_grad():
            for time in range(test_windows):
                window_input_set = np.concatenate([training_values, test_values[:, :time * horizon]],axis=1)
                input_set = TimeseriesSampler(timeseries=window_input_set, insample_size=input_size, outsample_size=horizon,
                                                window_sampling_limit=int(history_size * horizon))
                x, x_mask, x_timestamp, y_timestamp, ids = map(to_tensor, input_set.last_insample_window())
                x_z, y_z = PM(x_timestamp, ids.long(), all_Kmask), PM(y_timestamp, ids.long(), all_Kmask)
                window_z = y_z.cpu().detach().numpy()
                window_forecast, window_global, window_local = map(lambda x: x.cpu().detach().numpy(), EM(x, x_z, x_mask, y_z, ids))
                forecasts = window_forecast if len(forecasts) == 0 else np.concatenate([forecasts, window_forecast],axis=1)
                global_forecasts = window_global if len(global_forecasts) == 0 else np.concatenate([global_forecasts, window_global],axis=1)
                local_forecasts = window_local if len(local_forecasts) == 0 else np.concatenate([local_forecasts, window_local],axis=1)
                z_outputs = window_z if len(z_outputs) == 0 else np.concatenate([z_outputs, window_z],axis=1)
                # end for
        forecasts_df = pd.DataFrame(forecasts, columns=[f'V{t + 1}' for t in range(test_windows * horizon)]); forecasts_df.index.name = 'id'
        global_df = pd.DataFrame(global_forecasts, columns=[f'V{t + 1}' for t in range(test_windows * horizon)]); global_df.index.name = 'id'
        local_df = pd.DataFrame(local_forecasts, columns=[f'V{t + 1}' for t in range(test_windows * horizon)]); local_df.index.name = 'id'
        z_df = pd.DataFrame(z_outputs, columns=[f'V{t + 1}' for t in range(test_windows * horizon)]); z_df.index.name = 'id'
        # create dir
        if not os.path.isdir(OUTPUT_DIR):
            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        forecasts_df.to_csv(os.path.join(OUTPUT_DIR, f'forecast_{split_name}.csv'))
        global_df.to_csv(os.path.join(OUTPUT_DIR, f'global_{split_name}.csv'))
        local_df.to_csv(os.path.join(OUTPUT_DIR, f'local_{split_name}.csv'))
        z_df.to_csv(os.path.join(OUTPUT_DIR, f'z_{split_name}.csv'))
        #np.save(os.path.join(OUTPUT_DIR, f'alpha_{split_name}.npy'), EM.alpha.cpu().detach().numpy())
#np.save(os.path.join(OUTPUT_DIR, f'loss_{split_name}.npy'), np.array(training_loss_log))
#torch.save({'PM':PM.state_dict(),"EM":EM.state_dict()}, os.path.join(OUTPUT_DIR,f'model_{split_name}.pt'))
#np.save(os.path.join(OUTPUT_DIR, f'Kmask_{split_name}.npy'), all_Kmask.cpu().detach().numpy())
print("end", split_name)














