"""
All utility functions are defined here.
"""
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
    """
    Seed everything for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # training is extremely slow when do following setting
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
   

def get_device():
    """
    Get the device for training.
    """
    return ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu")
    

def generate_binomial_mask(B, T, C=None, p=0.5):
    """ Generate a binomial mask for a batch of data.
    Args:
        B (int): batch size
        T (int): length
        C (int): number of channels, if None, the mask will be the same for all channels
        p (float): probability to mask a timestamp
    Returns:
        res (torch.Tensor): a mask with shape (B, T, C) or (B, T)
    """
    if C:
        return torch.from_numpy(np.random.binomial(1, p, size=(B, T, C))).to(torch.bool)
    else:
        return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)
    
    
def generate_continuous_mask(B, T, C=None, n=5, l=0.1):
    """ Generate a continuous mask for a batch of data.
    Args:
        B (int): batch size
        T (int): length
        C (int): number of channels, if None, the mask will be the same for all channels
        n (int): number of masks
        l (float): length of each masks specified by a ratio of T
    Returns:
        res (torch.Tensor): a mask with shape (B, T, C) or (B, T)
    """
    if C:
        res = torch.full((B, T, C), True, dtype=torch.bool)
    else:
        res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            if C:
                # For a continuous timestamps, mask random half channels
                index = np.random.choice(C, int(C/2), replace=False)
                res[i, t:t + l, index] = False
            else:
                # For a continuous timestamps, mask all channels
                res[i, t:t+l] = False
    return res


def generate_ratio_mask(x, ratio=0.0):
    """
    Generate a mask for a batch of data by ratio.
    """
    mask = torch.rand(x.shape) > ratio # maskout_ratio are False
    mask = mask.to(x.device)
    return x * mask


class Logger(object):
    """ A Logger for saving output of printings between functions start_logging() and stop_logging().
    Args:
        filename (str): the name of the log file
    """
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
    """ Start logging the output of printings after calling this function.
    Args:
        seed (int): the random seed of this run, used for labeling the log file
        logdir (str): directory to save the log file
    """
    log_filename = f"log_{seed}.txt"
    log_filepath = os.path.join(logdir, log_filename)
    sys.stdout = Logger(log_filepath)


def stop_logging(logdir=None, seed=None, fraction=None, val_metrics=None, test_metrics=None):
    """ Stop logging the output of printings after calling this function.
    Args:
        logdir (str): directory to save the log file
        seed (int): seed for reproducibility
        fraction (float): fraction of the data used for training
        val_metrics (dict): validation metrics
        test_metrics (dict): test metrics
    """
    print() # print a blank line to separate
    sys.stdout = sys.__stdout__
    
    if val_metrics:
        val_path = os.path.join(logdir, f"val_{fraction}.csv")
        write_csv(val_path, seed, val_metrics)
        
    if test_metrics:
        test_path = os.path.join(logdir, f"test_{fraction}.csv")
        write_csv(test_path, seed, test_metrics)
        

def write_csv(path, seed, metrics):
    """ write metrics of different seeds to a csv file.
    Args:
        path (str): path to the csv file
        seed (int): random seed for this run
        metrics (dict): metrics to write
    """
    metrics = pd.DataFrame(metrics, index=[seed])
    # append to the existing csv file containing metrics of previous runs
    if os.path.isfile(path):
        df = pd.read_csv(path, index_col=0)
        df = pd.concat([df, metrics])
        df.to_csv(path)
    else:
        metrics.to_csv(path)
        
    
class MyBatchSampler(BatchSampler):
    """ 
    A custom BatchSampler to shuffle the samples within each batch.
    It changes the local order of samples(samples in the same batch) per epoch,
    which does not break too much the distribution of pre-shuffled samples by function shuffle_feature_label().
    The goal is to shuffle the samples per epoch but make sure that there are samples from the same trial in a batch.
    
    Args:
        sampler (Sampler): Base sampler
        batch_size (int): Size of mini-batch
        drop_last (bool): If True, the sampler will drop the last batch if its size would be less than batch_size
    """
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


def shuffle_feature_label(X, y, shuffle_function="trial", batch_size=128):
    """ Call shuffle functions.
    The goal is to guarantee that there are samples from the same trial in a batch,
    while avoiding all the samples are from the same trial/patient (low diversity).

    Args:
        shuffle_function (str): specify the shuffle function
        batch_size (int): batch_size if apply batch shuffle
    """

    # do trial shuffle
    if shuffle_function == "trial":
        return trial_shuffle_feature_label(X, y)

    # do batch shuffle
    elif shuffle_function == "batch":
        return batch_shuffle_feature_label(X, y, batch_size)

    # do random shuffle
    elif shuffle_function == "random":
        return shuffle(X, y)

    else:
        # print(shuffle_function)
        raise ValueError(f"\"{shuffle_function}\" is a wrong argument for shuffle function!")


def trial_shuffle_feature_label(X, y):
    """
    shuffle each samples in a trial first, then shuffle the order of trials
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


def batch_shuffle_feature_label(X, y, batch_size=256):
    """
    shuffle the order of batches first, then shuffle the samples in the batch
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


def freq_perturb(x, ratio=0.1):
    """
    Mask out some frequency components of the input signal by ratio.
    """
    xf = fft.rfft(x, dim=1)
    aug_1 = generate_ratio_mask(xf, ratio=ratio)
    xf = fft.irfft(aug_1, dim=1)
    return xf


# def add_frequency(x, ratio=0.0):
#     mask = torch.rand(x.shape) > (1 - ratio) # maskout_ratio are False
#     # mask = torch.cuda.FloatTensor(x.shape).uniform_() > (1-pertub_ratio) # only pertub_ratio of all values are True
#     mask = mask.to(x.device)
#     max_amplitude = x.max()
#     random_am = torch.rand(mask.shape) * (max_amplitude * 0.1)
#     pertub_matrix = mask * random_am
#     return x + pertub_matrix
    




  






        
    