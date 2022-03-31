import numpy as np
import torch
from dtaidistance import dtw
from tqdm import tqdm

def default_device() -> torch.device:
    """
    PyTorch default device is GPU when available, CPU otherwise.

    :return: Default device.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_tensor(array: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to tensor on default device.

    :param array: Numpy array to convert.
    :return: PyTorch tensor on default device.
    """
    return torch.tensor(array, dtype=torch.float32).to(default_device())

def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result

def group_values(values: np.ndarray, groups: np.ndarray, group_name: str) -> np.ndarray:
    """
    Filter values array by group indices and clean it from NaNs.

    :param values: Values to filter.
    :param groups: Timeseries groups.
    :param group_name: Group name to filter by.
    :return: Filtered and cleaned timeseries.
    """
    return np.array([v[~np.isnan(v)] for v in values[groups == group_name]])


"""
FFT for Periodic Module
"""
def FFT(current_series, current_mean):
    # since some series might begin with many 0s, we get <nonzero_idx> 
    # in order to get started with non-zero part of series
    if np.sum(current_series==-current_mean) > 0:
        try:
            nonzero_idx = np.where(current_series > 0)[0][0]
        except:
            nonzero_idx = 0
    else:
        nonzero_idx = 0
    series = current_series[nonzero_idx:]
    # real valid data for FFT
    N = len(series)
    t = np.arange(N)
    dt = 1
    # transform
    fft = np.fft.fft(series)
    fftshift = np.fft.fftshift(fft)
    mo = abs(fftshift) / N
    phase = np.angle(fftshift)
    fre = np.fft.fftshift(np.fft.fftfreq(d = dt, n = N))
    # series will shift 2*pi*f*nonzeros_idx
    apfs = [(mo[i], phase[i]-2*np.pi*fre[i]*nonzero_idx, fre[i]) for i in range(len(mo))]
    return apfs


def get_bestKmask_per_series(K_apfs, series, fft_cutpoint, J):
    # we use dtw for matching and selection
    criterion = dtw.distance_fast
    bestKsubset = []
    fftinput = np.arange(len(series))
    recover_results = np.zeros(len(series))
    bestdtw = criterion(series[fft_cutpoint:], recover_results[fft_cutpoint:] )
    for i,item in enumerate(K_apfs):
        a,p,f = item
        current_fft =  a*np.cos(2*np.pi*f*fftinput + p)
        current_dtw = criterion(series[fft_cutpoint:], recover_results[fft_cutpoint:]+current_fft[fft_cutpoint:])
        if current_dtw < bestdtw:
            bestKsubset.append(i)
            bestdtw = current_dtw
            recover_results = recover_results + current_fft
            if len(bestKsubset) >= J:
                break
    # transform bestKsubset into one-hot mask
    selected_results = np.zeros(len(K_apfs))
    for idx in bestKsubset:
        selected_results[idx] = 1
    return selected_results


"""
FFT for Periodic Module per K for each series
"""
def warm_PM_parameters_perK(training_values, fft_cutpoint, K=100, J=10):
    # compute for initializaion of periodical module
    all_f, all_p, all_a = [], [], []
    all_mean = []
    all_Kmask = []
    for i, current_series in tqdm(enumerate(training_values)):
        all_mean.append(np.mean(current_series))
        current_mean = np.mean(current_series)
        series = current_series - all_mean[-1]
        fftnet_series = series[:fft_cutpoint]
        apfs = FFT(fftnet_series, current_mean)
        sorted_apfs = sorted(apfs, key=lambda x: x[0], reverse=True)
        K_apfs = []
        # remove the same source infomation
        for item in sorted_apfs:
            a,p,f = item
            if len(K_apfs) == 0:
                K_apfs.append(item)
                continue
            if len(K_apfs) == K:
                break
            # since cos(x) = cos(-x), we want to simplify the candidate cos functions of equivalent p and f
            if round(p + K_apfs[-1][1],4) == 0.0 and round(f + K_apfs[-1][2],4) == 0.0:
                K_apfs[-1] = (K_apfs[-1][0] + a, K_apfs[-1][1], K_apfs[-1][2])
            else:
                K_apfs.append(item)
        # keep all series have K dimensions source cos
        if len(K_apfs) < K:
            for _ in range(K-len(K_apfs)):
                K_apfs.append((0,0,0))
        # get bestK for per series
        Kmask = get_bestKmask_per_series(K_apfs, series, fft_cutpoint, J)
        K_a = [item[0] for item in K_apfs]
        K_p = [item[1] for item in K_apfs]
        K_f = [item[2] for item in K_apfs]
        all_f.append(K_f)
        all_a.append(K_a)
        all_p.append(K_p)
        all_Kmask.append(Kmask)
    return np.array(all_a), np.array(all_p), np.array(all_f), np.array(all_mean), np.array(all_Kmask)




# Adopt from https://github.com/ElementAI/N-BEATS
"""
Timeseries sampler
"""
import numpy as np

class TimeseriesSampler:
    def __init__(self,
                 timeseries,
                 insample_size: int,
                 outsample_size: int,
                 window_sampling_limit: int,
                 batch_size: int = 1024):
        """
        Timeseries sampler.

        :param timeseries: Timeseries data to sample from. Shape: timeseries, timesteps
        :param insample_size: Insample window size. If timeseries is shorter then it will be 0-padded and masked.
        :param outsample_size: Outsample window size. If timeseries is shorter then it will be 0-padded and masked.
        :param window_sampling_limit: Size of history the sampler should use.
        :param batch_size: Number of sampled windows.
        """
        self.timeseries = [ts for ts in timeseries]
        self.ids = list(range(len(self.timeseries)))
        self.window_sampling_limit = window_sampling_limit
        self.batch_size = batch_size
        self.insample_size = insample_size
        self.outsample_size = outsample_size

    def __iter__(self):
        """
        Batches of sampled windows.

        :return: Batches of:
         Insample: "batch size, insample size"
         Insample mask: "batch size, insample size"
         Outsample: "batch size, outsample size"
         Outsample mask: "batch size, outsample size"
        """
        while True:
            insample = np.zeros((self.batch_size, self.insample_size))
            insample_mask = np.zeros((self.batch_size, self.insample_size))
            insample_timestamp = np.zeros((self.batch_size, self.insample_size))
            outsample = np.zeros((self.batch_size, self.outsample_size))
            outsample_mask = np.zeros((self.batch_size, self.outsample_size))
            outsample_timestamp = np.zeros((self.batch_size, self.outsample_size))
            sampled_ts_indices = np.random.randint(len(self.timeseries), size=self.batch_size) # series_id
            for i, sampled_index in enumerate(sampled_ts_indices):
                sampled_timeseries = self.timeseries[sampled_index]
                cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                              high=len(sampled_timeseries),
                                              size=1)[0]

                insample_window = sampled_timeseries[max(0, cut_point - self.insample_size):cut_point]
                insample_timewindow = np.arange(max(0, cut_point - self.insample_size), cut_point)
                insample[i, -len(insample_window):] = insample_window
                insample_mask[i, -len(insample_window):] = 1.0
                insample_timestamp[i, -len(insample_timewindow):] = insample_timewindow

                outsample_window = sampled_timeseries[
                                   cut_point:min(len(sampled_timeseries), cut_point + self.outsample_size)]
                outsample_timewindow = np.arange(cut_point, min(len(sampled_timeseries), cut_point + self.outsample_size))
                outsample[i, :len(outsample_window)] = outsample_window
                outsample_mask[i, :len(outsample_window)] = 1.0
                outsample_timestamp[i, :len(outsample_timewindow)] = outsample_timewindow
            yield insample, insample_mask, outsample, outsample_mask, insample_timestamp, outsample_timestamp, sampled_ts_indices # series_id

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.insample_size))
        insample_mask = np.zeros((len(self.timeseries), self.insample_size))
        insample_timestamp = np.zeros((len(self.timeseries), self.insample_size))
        outsample_timestamp = np.zeros((len(self.timeseries), self.outsample_size))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.insample_size:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
            ts_last_timewindow = np.arange(len(ts)-self.insample_size ,len(ts))
            insample_timestamp[i, -len(ts_last_timewindow):] = ts_last_timewindow
            ts_next_timewindow = np.arange(len(ts), len(ts) + self.outsample_size)
            outsample_timestamp[i, -len(ts_next_timewindow):] = ts_next_timewindow
        return insample, insample_mask, insample_timestamp, outsample_timestamp, np.arange(len(self.timeseries)) # series_id



# Adopt from https://github.com/ElementAI/N-BEATS
"""
Loss functions for PyTorch.
"""

def mape_loss(forecast: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
    """
    MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :param mask: 0/1 mask. Shape: batch, time
    :return: Loss value
    """
    weights = divide_no_nan(mask, target)
    return torch.mean(torch.abs((forecast - target) * weights))

def smape_1_loss(forecast: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
    """
    sMAPE loss as defined in "Appendix A" of
    http://www.forecastingprinciples.com/files/pdf/Makridakia-The%20M3%20Competition.pdf

    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :param mask: 0/1 mask. Shape: batch, time
    :return: Loss value
    """
    return 200 * torch.mean(divide_no_nan(torch.abs(forecast - target), forecast.data + target.data + 1e-8) * mask)


def smape_2_loss(forecast, target, mask) -> torch.float:
    """
    sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

    :param forecast: Forecast values. Shape: batch, time
    :param target: Target values. Shape: batch, time
    :param mask: 0/1 mask. Shape: batch, time
    :return: Loss value
    """
    return 200 * torch.mean(divide_no_nan(torch.abs(forecast - target),
                                      torch.abs(forecast.data) + torch.abs(target.data) + 1e-8) * mask)


def mase_loss(insample: torch.Tensor, freq: int,
              forecast: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
    """
    MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

    :param insample: Insample values. Shape: batch, time_i
    :param freq: Frequency value
    :param forecast: Forecast values. Shape: batch, time_o
    :param target: Target values. Shape: batch, time_o
    :param mask: 0/1 mask. Shape: batch, time_o
    :return: Loss value
    """
    masep = torch.mean(torch.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
    masked_masep_inv = divide_no_nan(mask, masep[:, None] + 1e-8)
    return torch.mean(torch.abs(target - forecast) * masked_masep_inv)


