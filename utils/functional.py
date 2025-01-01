import itertools
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


def set_seed(seed):
    '''
    set the random seed
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

def get_device():
    '''
    get device
    '''
    return (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.mps.is_available()
        else 'cpu'
        )
    
    
def save_checkpoint(checkpoint, is_best, path):
    '''
    save the model
    '''
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))    
    torch.save(checkpoint, path)
    if is_best:
        shutil.copyfile(path, os.path.join(os.path.dirname(path), 'best.pth'))
        
    
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


def create_views(X, y, nviews=2, flatten=False):
    '''
    create the views for the contrastive learning
    '''
    nchannels = X.shape[-1]
    
    X = X.reshape(-1, nviews, X.shape[1], X.shape[2])
    y = y.reshape(-1, nviews, y.shape[1])
    
    if flatten:
        X = X.transpose(3, 0, 1, 2).reshape(-1, nviews, X.shape[1], 1)
        y = np.tile(y, (nchannels, 1, 1))
        
    return X, y
        
    