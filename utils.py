import itertools
import sys
import torch
import random
import os
import shutil
import numpy as np
import pandas as pd
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from scipy import interpolate
from torch.utils.data import BatchSampler


def seed_everything(seed=42):
    """
    Seed everything.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # training is extremely slow when do following setting
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
   
    
class Logger(object):
    """ A Logger for saving output of printings between functions start_logging() and stop_logging().

    """
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)


def start_logging(random_seed, saving_directory):
    log_filename = f"log_{random_seed}.txt"
    log_filepath = os.path.join(saving_directory, log_filename)
    sys.stdout = Logger(log_filepath)


def stop_logging():
    sys.stdout = sys.__stdout__


class MyBatchSampler(BatchSampler):
    """ A custom BatchSampler to shuffle the samples within each batch.
        It changes the local order of samples(samples in the same batch) per epoch,
        which does not break too much the distribution of pre-shuffled samples by function shuffle_feature_label().
        The goal is to shuffle the samples per epoch but make sure that there are samples from the same trial in a batch.

    """
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                random.shuffle(batch)
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            random.shuffle(batch)
            yield batch


def shuffle_feature_label(X, y, shuffle_function='trial', batch_size=128):
    """ Call shuffle functions.
        The goal is to guarantee that there are samples from the same trial in a batch,
        while avoiding all the samples are from the same trial/patient (low diversity).

    Args:
        shuffle_function (str): specify the shuffle function
        batch_size (int): batch_size if apply batch shuffle
    """

    # do trial shuffle
    if shuffle_function == 'trial':
        return trial_shuffle_feature_label(X, y)

    # do batch shuffle
    elif shuffle_function == 'batch':
        return batch_shuffle_feature_label(X, y, batch_size)

    # do random shuffle
    elif shuffle_function == 'random':
        return shuffle(X, y, random_state=42)

    else:
        # print(shuffle_function)
        raise ValueError(f'\'{shuffle_function}\' is a wrong argument for shuffle function!')


def trial_shuffle_feature_label(X, y):
    """ shuffle each samples in a trial first, then shuffle the order of trials

    """

    # sort X, y by trial ID
    sorted_indices = np.argsort(y[:, 2], axis=0)
    # concatenate sorted indices and labels
    sorted_indices_labels = np.concatenate((sorted_indices.reshape(-1, 1), y[sorted_indices]), axis=1).astype(int)
    trials_list = []
    # group each trial by trial ID
    for _, trial in itertools.groupby(sorted_indices_labels, lambda x: x[3]):
        trial = list(trial)
        # shuffle each sample in a trial
        trial = shuffle(trial, random_state=42)
        trials_list.append(trial)
    # shuffle the order of trials
    shuffled_trials_list = shuffle(trials_list, random_state=42)
    shuffled_trials = np.concatenate(shuffled_trials_list, axis=0)
    # get the sorted indices
    shuffled_sorted_indices = shuffled_trials[:, 0]
    X_shuffled = X[shuffled_sorted_indices]
    y_shuffled = y[shuffled_sorted_indices]
    return X_shuffled, y_shuffled


def batch_shuffle_feature_label(X, y, batch_size=128):
    """ shuffle the order of batches first, then shuffle the samples in the batch

    """

    # sort X, y by trial ID
    sorted_indices = np.argsort(y[:, 2], axis=0)
    sorted_indices_list = np.array_split(sorted_indices, y.shape[0]/batch_size)
    # shuffle the batches
    sorted_indices_list = shuffle(sorted_indices_list, random_state=42)
    # shuffle samples in the batch
    shuffled_sorted_indices_list = []
    for batch in sorted_indices_list:
        shuffled_batch = shuffle(batch, random_state=42)
        shuffled_sorted_indices_list.append(shuffled_batch)
    shuffled_sorted_indices = np.concatenate(shuffled_sorted_indices_list, axis=0)
    X_shuffled = X[shuffled_sorted_indices]
    y_shuffled = y[shuffled_sorted_indices]
    return X_shuffled, y_shuffled

  
def resample(data, freq1=500, freq2=250, kind='linear'):
    '''
    resample the data from freq1 to freq2
    '''
    t = np.linspace(1, len(data), len(data))
    f = interpolate.interp1d(t, data, kind=kind)
    t_new = np.linspace(1, len(data), int(len(data)/freq1 * freq2))
    new_data = f(t_new)
    return new_data

 
def normalize(data):
    '''
    normalize the data by x=x-mean/std
    '''
    scaler = StandardScaler()
    norm_data = scaler.fit_transform(data)
    return norm_data


def R_Peaks(ecg):
    '''
    extract the R Peaks from the ECG signal
    '''
    # get R Peak positions
    pos = []
    # get R Peak intervals
    trial_interval = []
    for ch in range(ecg.shape[1]):
        cleaned_ecg = nk.ecg_clean(ecg[:, ch], sampling_rate=250, method='neurokit')
        signals, _ = nk.ecg_peaks(cleaned_ecg, sampling_rate=250, correct_artifacts=False)
        peaks = signals[signals['ECG_R_Peaks']==1].index.to_list()
        pos.append(peaks)
        channel_interval = []
        for i in range(len(peaks)-1):
            channel_interval.append(peaks[i+1] - peaks[i])
        trial_interval.append(channel_interval)
        
    df_peaks = pd.DataFrame(pos) # [num of the R-Peaks of a channel]
    df = pd.DataFrame(trial_interval).T
    med = df.median()
    return df, med, df_peaks
    
    
def trial2sample(data, max_duration=300):
    '''
    split resampled trial to sample level(single heartbeat)
    '''
    samples = []
    _, med, df_peaks = R_Peaks(data)
    trial_med = med.median()
    for i in range(df_peaks.shape[1]):
        RP_pos = df_peaks.iloc[:, i].median()
        beat = data[max(0,int(RP_pos)-int(trial_med/2)):min(int(RP_pos)+int(trial_med/2),data.shape[0]), :]
        left_zero_num = int((int(max_duration)-beat.shape[0])/2)
        padding_left = np.zeros((left_zero_num, data.shape[1]))
        padding_right = np.zeros((int(max_duration)-left_zero_num-beat.shape[0], data.shape[1]))
        beat = np.concatenate([padding_left, beat, padding_right], axis=0)
        samples.append(beat)
    return samples 


def sample2trial(samples, size=10):
    '''
    concat samples to pseudo-trials
    '''
    trials = []
    idx = 0
    while idx <= len(samples)-size:
        beat = samples[idx]
        for i in range(idx+1, idx+size):
            beat = np.vstack((beat, samples[i]))
        trials.append(beat)
        idx += size
    return trials


def segment(X, y, sample):
    '''
    segment the trial to non-overlapping samples
    '''
    length = X.shape[1]
    assert length % sample == 0
    nsample = length / sample
    
    samples = X.reshape(-1, sample, X.shape[-1])
    tids = np.repeat(np.arange(y.shape[0])+1, nsample)
    labels = np.repeat(y, nsample, axis=0)
    labels = np.hstack([labels, tids.reshape(labels.shape[0], -1)])
    return samples, labels


def trial_shuffle(X, y):
    '''
    shuffle the data by the intra-inter trial procedure
    '''
    # sort X, y by trial ID
    sorted_indices = np.argsort(y[:, 2], axis=0)
    # concatenate sorted indices and labels
    sorted_indices_labels = np.concatenate((sorted_indices.reshape(-1, 1), y[sorted_indices]), axis=1).astype(int)
    trials_list = []
    # group each trial by trial ID
    for _, trials in itertools.groupby(sorted_indices_labels, lambda x: x[3]):
        trials = list(trials)
        # shuffle each sample in a trial
        trials = shuffle(trials)
        trials_list.append(trials)
    # shuffle the order of trials
    shuffled_trials_list = shuffle(trials_list)
    shuffled_trials = np.concatenate(shuffled_trials_list, axis=0)
    # get the sorted indices
    shuffled_sorted_indices = shuffled_trials[:, 0]
    X_shuffled = X[shuffled_sorted_indices]
    y_shuffled = y[shuffled_sorted_indices]
    return X_shuffled, y_shuffled
        
    