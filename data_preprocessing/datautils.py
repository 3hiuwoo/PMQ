import numpy as np
import pandas as pd
import neurokit2 as nk
import torch.nn.functional as F
from scipy import interpolate
from sklearn.preprocessing import StandardScaler

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