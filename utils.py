''' Utilize COMET code from from:
    https://github.com/DL4mHealth/COMET/blob/main/utils.py
    https://github.com/DL4mHealth/COMET/blob/main/data_preprocessing/PTB/PTB_preprocessing.ipynb
'''
import itertools
import torch
import random
import os
import sys
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import fft
from sklearn.utils import shuffle
from torch.utils.data import BatchSampler

def seed_everything(seed=42):
    '''
    Seed everything for reproducibility.
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # training is extremely slow when do following setting
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
   

def get_device():
    '''
    Get the device for training.
    '''
    return ('cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu')
    
    
class Logger(object):
    ''' A Logger for saving output of printings between functions start_logging() and stop_logging().

    '''
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")


    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
      
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()


def start_logging(seed, logdir):
    log_filename = f"log_{seed}.txt"
    log_filepath = os.path.join(logdir, log_filename)
    sys.stdout = Logger(log_filepath)


def stop_logging(logdir=None, seed=None, fraction=None, val_metrics=None, test_metrics=None):
    print()
    sys.stdout = sys.__stdout__
    
    if val_metrics:
        val_path = os.path.join(logdir, f'val_{fraction}.csv')
        write_csv(val_path, seed, val_metrics)
        
    if test_metrics:
        test_path = os.path.join(logdir, f'test_{fraction}.csv')
        write_csv(test_path, seed, test_metrics)
        

def write_csv(path, seed, metrics):
    metrics = pd.DataFrame(metrics, index=[seed])
    if os.path.isfile(path):
        df = pd.read_csv(path, index_col=0)
        df = pd.concat([df, metrics])
        df.to_csv(path)
    else:
        metrics.to_csv(path)
        
    
class MyBatchSampler(BatchSampler):
    ''' 
    A custom BatchSampler to shuffle the samples within each batch.
    It changes the local order of samples(samples in the same batch) per epoch,
    which does not break too much the distribution of pre-shuffled samples by function shuffle_feature_label().
    The goal is to shuffle the samples per epoch but make sure that there are samples from the same trial in a batch.
    '''
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    random.shuffle(batch)
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    random.shuffle(batch)
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]


def shuffle_feature_label(X, y, shuffle_function='trial', batch_size=128):
    '''Call shuffle functions.
    The goal is to guarantee that there are samples from the same trial in a batch,
    while avoiding all the samples are from the same trial/patient (low diversity).

    Args:
        shuffle_function (str): specify the shuffle function
        batch_size (int): batch_size if apply batch shuffle
    '''

    # do trial shuffle
    if shuffle_function == 'trial':
        return trial_shuffle_feature_label(X, y)

    # do batch shuffle
    elif shuffle_function == 'batch':
        return batch_shuffle_feature_label(X, y, batch_size)

    # do random shuffle
    elif shuffle_function == 'random':
        return shuffle(X, y)

    else:
        # print(shuffle_function)
        raise ValueError(f'\'{shuffle_function}\' is a wrong argument for shuffle function!')


def trial_shuffle_feature_label(X, y):
    '''
    shuffle each samples in a trial first, then shuffle the order of trials
    '''
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


def batch_shuffle_feature_label(X, y, batch_size=256):
    '''
    shuffle the order of batches first, then shuffle the samples in the batch
    '''

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


def transform(x, opt='t'):
    if opt[0] == 't':
        re = x
    elif opt[0] == 'f':
        re = freq_perturb(x, ratio=0.1)
    elif opt[0] == 's':
        re = fft_interp(x, target_len=300)
    
    if len(opt) == 1:
        mask = 'all_true'
    elif opt[1:] == 'b':
        mask = 'binomial'
    elif opt[1:] == 'c':
        mask = 'continuous'
    elif opt[1:] == 'cb':
        mask = 'channel_binomial'
    elif opt[1:] == 'cc':
        mask = 'channel_continuous'
    
    return re, mask
    
        

def freq_perturb(x, ratio=0.1):
    xf = fft.rfft(x, dim=1)
    aug_1 = remove_frequency(xf, ratio=ratio)
    xf = fft.irfft(aug_1, dim=1)
    # aug_2 = add_frequency(x, ratio=ratio)
    # aug_F = aug_1 + aug_2
    # return aug_F
    return xf


def remove_frequency(x, ratio=0.0):
    mask = torch.rand(x.shape) > ratio # maskout_ratio are False
    # mask = torch.cuda.FloatTensor(x.shape).uniform_() > pertub_ratio # maskout_ratio are False
    mask = mask.to(x.device)
    return x * mask


def add_frequency(x, ratio=0.0):
    mask = torch.rand(x.shape) > (1 - ratio) # maskout_ratio are False
    # mask = torch.cuda.FloatTensor(x.shape).uniform_() > (1-pertub_ratio) # only pertub_ratio of all values are True
    mask = mask.to(x.device)
    max_amplitude = x.max()
    random_am = torch.rand(mask.shape) * (max_amplitude * 0.1)
    pertub_matrix = mask * random_am
    return x + pertub_matrix 


def fft_interp(x, target_len):
    '''
    Interpolate the time-series data to the target length.
    '''
    xf = fft.rfft(x, dim=1).abs()
    xf = xf.permute(0, 2, 1)
    xf = F.interpolate(xf, size=target_len, mode='linear')
    return xf  
  






        
    