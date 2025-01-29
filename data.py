import os
import numpy as np
from tqdm import tqdm
from utils import split_data_label, process_batch_ts
from sklearn.utils import shuffle
    
def load_data(root='dataset', name='chapman', length=None, overlap=0, norm=True, shuff=True):
    '''
    load and preprocess data
    '''
    data_path = os.path.join(root, name, 'feature')
    labels, train_ids, valid_ids, test_ids = load_label_split(root, name)
    
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
    
    for i, fn in enumerate(tqdm(filenames, desc=f'=> Loading {name}')):
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
      
    if length:
        # X_train, y_train = segment(X_train, y_train, split)
        # X_val, y_val = segment(X_val, y_val, split)
        # X_test, y_test = segment(X_test, y_test, split)
        
        X_train, y_train = split_data_label(X_train, y_train, sample_timestamps=length, overlapping=overlap)
        X_val, y_val = split_data_label(X_val, y_val, sample_timestamps=length, overlapping=overlap)
        X_test, y_test = split_data_label(X_test, y_test, sample_timestamps=length, overlapping=overlap)
        
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_label_split(root='dataset', name='chapman'):
    '''
    load labels for dataset and split information
    '''
    label_path = os.path.join(root, name, 'label', 'label.npy')
    labels = np.load(label_path)
    
    if name == 'chapman':
        pids_sb = list(labels[np.where(labels[:, 0]==0)][:, 1])
        pids_af = list(labels[np.where(labels[:, 0]==1)][:, 1])
        pids_gsvt = list(labels[np.where(labels[:, 0]==2)][:, 1])
        pids_sr = list(labels[np.where(labels[:, 0]==3)][:, 1])
        
        train_ids = pids_sb[:-500] + pids_af[:-500] + pids_gsvt[:-500] + pids_sr[:-500]
        val_ids = pids_sb[-500:-250] + pids_af[-500:-250] + pids_gsvt[-500:-250] + pids_sr[-500:-250]
        test_ids = pids_sb[-250:] + pids_af[-250:] + pids_gsvt[-250:] + pids_sr[-250:]
        
    elif name == 'ptb':
        pids_neg = list(labels[np.where(labels[:, 0]==0)][:, 1])
        pids_pos = list(labels[np.where(labels[:, 0]==1)][:, 1])
        
        train_ids = pids_neg[:-14] + pids_pos[:-42]  # specify patient ID for training, validation, and test set
        val_ids = pids_neg[-14:-7] + pids_pos[-42:-21]   # 28 patients, 7 healthy and 21 positive
        test_ids = pids_neg[-7:] + pids_pos[-21:]  # # 28 patients, 7 healthy and 21 positive
        
    elif name == 'ptbxl':
        pids_norm = list(labels[np.where(labels[:, 0]==0)][:, 1])
        pids_mi = list(labels[np.where(labels[:, 0]==1)][:, 1])
        pids_sttc = list(labels[np.where(labels[:, 0]==2)][:, 1])
        pids_cd = list(labels[np.where(labels[:, 0]==3)][:, 1])
        pids_hyp = list(labels[np.where(labels[:, 0]==3)][:, 1])
        
        train_ids = pids_norm[:-1200] + pids_mi[:-400] + pids_sttc[:-400] + pids_cd[:-200] + pids_hyp[:-200]
        val_ids = pids_norm[-1200:-600] + pids_mi[-400:-200] + pids_sttc[-400:-200] + pids_cd[-200:-100] + pids_hyp[-200:-100]
        test_ids = pids_norm[-600:] + pids_mi[-200:] + pids_sttc[-200:] + pids_cd[-100:] + pids_hyp[-100:]
        
    else:
        raise ValueError(f'Unknown dataset: {name}')
        
    return labels, train_ids, val_ids, test_ids
        
