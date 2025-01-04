import os
import numpy as np
from tqdm import tqdm
from utils import segment
from sklearn.utils import shuffle
from utils import process_batch_ts


def load_data(root='data', name='chapman', split=None, norm=True, shuffle=True):
    '''
    load correspondent training, validation, and test data and labels
    '''
    if name == 'chapman':
        dir = os.path.join(root, name)
        return load_chapman(dir, split, norm, shuffle)
    else:
        raise ValueError(f'Unknown dataset name: {name}')
    
    
def load_chapman(root='data/chapman', split=None, norm=True, shuff=True):
    data_path = os.path.join(root, 'feature')
    label_path = os.path.join(root, 'label', 'label.npy')
    
    labels = np.load(label_path)
    
    pids_sb = list(labels[np.where(labels[:, 0]==0)][:, 1])
    pids_af = list(labels[np.where(labels[:, 0]==1)][:, 1])
    pids_gsvt = list(labels[np.where(labels[:, 0]==2)][:, 1])
    pids_sr = list(labels[np.where(labels[:, 0]==3)][:, 1])
    
    train_ids = pids_sb[:-500] + pids_af[:-500] + pids_gsvt[:-500] + pids_sr[:-500]
    valid_ids = pids_sb[-500:-250] + pids_af[-500:-250] + pids_gsvt[-500:-250] + pids_sr[-500:-250]
    test_ids = pids_sb[-250:] + pids_af[-250:] + pids_gsvt[-250:] + pids_sr[-250:]
    
    filenames = []
    for fn in os.listdir(data_path):
        filenames.append(fn)
    filenames.sort()
    
    train_trials = []
    train_labels = []
    valid_trials = []
    valid_labels = []
    test_trials = []
    test_labels = []
    
    for i, fn in enumerate(tqdm(filenames, desc=f'=> Loading Chapman')):
        label = labels[i]
        feature = np.load(os.path.join(data_path, fn))
        for trial in feature:
            if i+1 in train_ids:
                train_trials.append(trial)
                train_labels.append(label)
            elif i+1 in valid_ids:
                valid_trials.append(trial)
                valid_labels.append(label)
            elif i+1 in test_ids:
                test_trials.append(trial)
                test_labels.append(label)
                
    X_train = np.array(train_trials)
    X_val = np.array(valid_trials)
    X_test = np.array(test_trials)
    y_train = np.array(train_labels)
    y_val = np.array(valid_labels)
    y_test = np.array(test_labels)
    
    if shuff:
        X_train, y_train = shuffle(X_train, y_train)
        X_val, y_val = shuffle(X_val, y_val)
        X_test, y_test = shuffle(X_test, y_test)
    
    if norm:
        X_train = process_batch_ts(X_train, normalized=True, bandpass_filter=False)
        X_val = process_batch_ts(X_val, normalized=True, bandpass_filter=False)
        X_test = process_batch_ts(X_test, normalized=True, bandpass_filter=False)
      
    if split:
        X_train, y_train = segment(X_train, y_train, split)
        X_val, y_val = segment(X_val, y_val, split)
        X_test, y_test = segment(X_test, y_test, split)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
