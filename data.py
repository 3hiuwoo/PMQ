"""
This file contains functions for loading and preprocessing datasets, e.g. segment, normalize, shuffle.
You might always import only load_data from this file since other files are sub-functions of load_data.
Please construct the dataset of our format following the README file.
Supporting dataset includes chapman, ptb, ptbxl, cpsc2018, currently.
If want to add more datasets, you may construct the dataset and implement splits information in load_split_ids().
"""
import os
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter
from itertools import repeat
    
def load_data(root="dataset", name="chapman", length=None, overlap=0, norm=True, neighbor=False):
    """ load, segment and preprocess train/val/test data
    
    Args:
        root (str): root path for directory containing datasets folder
        name (str): dataset name
        length (int): segment length, if None, no segment
        overlap (float): segment overlapping ratio
        norm (bool): whether to normalize the data
        neighbor (bool): whether to split the data into two halves
        
    Returns:
        X_train (numpy.ndarray): train data
        X_val (numpy.ndarray): validation data
        X_test (numpy.ndarray): test data
        y_train (numpy.ndarray): train labels
        y_val (numpy.ndarray): validation labels
        y_test (numpy.ndarray): test labels
    """
    data_path = os.path.join(root, name, "feature")
    labels, train_ids, valid_ids, test_ids = load_split_ids(root, name)
    
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
    
    for i, fn in enumerate(tqdm(filenames, desc=f"=> Loading {name}")):
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
    
    # X_train, y_train = shuffle(X_train, y_train, random_state=42)
    # X_val, y_val = shuffle(X_val, y_val, random_state=42)
    # X_test, y_test = shuffle(X_test, y_test, random_state=42)

    # only use first 12 leads for PTB dataset, which is the same as other datasets
    if name == "ptb":
        X_train = X_train[:, :, :12]
        X_val = X_val[:, :, :12]
        X_test = X_test[:, :, :12]

    if norm:
        X_train = process_batch_ts(X_train, normalized=True, bandpass_filter=False)
        X_val = process_batch_ts(X_val, normalized=True, bandpass_filter=False)
        X_test = process_batch_ts(X_test, normalized=True, bandpass_filter=False)
      
    if length:
        X_train, y_train = split_data_label(X_train, y_train, sample_timestamps=length, overlapping=overlap)
        X_val, y_val = split_data_label(X_val, y_val, sample_timestamps=length, overlapping=overlap, keep_dim=True)
        X_test, y_test = split_data_label(X_test, y_test, sample_timestamps=length, overlapping=overlap, keep_dim=True)
        
    if neighbor:
        length = X_train.shape[1]
        nleads = X_train.shape[-1]
        assert length % 2 == 0
        
        X_train = X_train.reshape(-1, 2, int(length/2), nleads)
        
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_split_ids(root="dataset", name="chapman"):
    """ split patients into train, validation, and test set for loading data
    
    Args:
        root (str): root path for directory containing datasets folder
        name (str): dataset name
        
    Returns:
        labels (numpy.ndarray): all labels
        train_ids (list): patient IDs for training set
        val_ids (list): patient IDs for validation set
        test_ids (list): patient IDs
    """
    label_path = os.path.join(root, name, "label", "label.npy")
    labels = np.load(label_path)
    
    if name == "chapman":
        # 4 classes: SB, AF, GSVT, SR
        pids_sb = list(labels[np.where(labels[:, 0]==0)][:, 1])
        pids_af = list(labels[np.where(labels[:, 0]==1)][:, 1])
        pids_gsvt = list(labels[np.where(labels[:, 0]==2)][:, 1])
        pids_sr = list(labels[np.where(labels[:, 0]==3)][:, 1])
        
        print(f"=> Dataset {name} has {len(labels)} patients, with {len(pids_sb)} SB, {len(pids_af)} AF, {len(pids_gsvt)} GSVT, {len(pids_sr)} SR")
        
        idxs = np.rint([0.1*len(pids_sb), 0.1*len(pids_af), 0.1*len(pids_gsvt), 0.1*len(pids_sr)]).astype(int)
        
        train_ids = pids_sb[:-2*idxs[0]] + pids_af[:-2*idxs[1]] + pids_gsvt[:-2*idxs[2]] + pids_sr[:-2*idxs[3]]
        val_ids = pids_sb[-2*idxs[0]:-idxs[0]] + pids_af[-2*idxs[1]:-idxs[1]] + pids_gsvt[-2*idxs[2]:-idxs[2]] + pids_sr[-2*idxs[3]:-idxs[3]]
        test_ids = pids_sb[-idxs[0]:] + pids_af[-idxs[1]:] + pids_gsvt[-idxs[2]:] + pids_sr[-idxs[3]:]
         
    elif name == "ptb":
        # 2 classes: healthy, MI
        pids_neg = list(labels[np.where(labels[:, 0]==0)][:, 1])
        pids_pos = list(labels[np.where(labels[:, 0]==1)][:, 1])
        
        print(f"=> Dataset {name} has {len(labels)} patients, with {len(pids_neg)} healthy, {len(pids_pos)} MI")
        
        train_ids = pids_neg[:-14] + pids_pos[:-42]
        val_ids = pids_neg[-14:-7] + pids_pos[-42:-21]   # 28 patients, 7 healthy and 21 positive
        test_ids = pids_neg[-7:] + pids_pos[-21:]  # # 28 patients, 7 healthy and 21 positive
        
    elif name == "ptbxl":
        # 5 classes: normal, MI, STTC, CD, HYP
        pids_norm = list(labels[np.where(labels[:, 0]==0)][:, 1])
        pids_mi = list(labels[np.where(labels[:, 0]==1)][:, 1])
        pids_sttc = list(labels[np.where(labels[:, 0]==2)][:, 1])
        pids_cd = list(labels[np.where(labels[:, 0]==3)][:, 1])
        pids_hyp = list(labels[np.where(labels[:, 0]==3)][:, 1])
        
        print(f"=> Dataset {name} has {len(labels)} patients, with {len(pids_norm)} normal, {len(pids_mi)} MI, {len(pids_sttc)} STTC, {len(pids_cd)} CD, {len(pids_hyp)} HYP")
        
        folds = np.load(os.path.join(root, name, "label", "fold.npy"))
        
        train_ids = list(folds[np.where((folds[:, 0] != 9) & (folds[:, 0] != 10))][:, 1])
        val_ids = list(folds[np.where(folds[:, 0] == 9)][:, 1])
        test_ids = list(folds[np.where(folds[:, 0] == 10)][:, 1])
        
    elif name == "cpsc2018":
        # 9 classes: Normal, AF, IAVB, LBBB, RBBB, PAC, PVC, STD, STE
        labels[:, 0] -= 1 # original labels start from 1
        
        pids_Normal = list(labels[np.where(labels[:, 0] == 0)][:, 1])
        pids_AF = list(labels[np.where(labels[:, 0] == 1)][:, 1])
        pids_IAVB = list(labels[np.where(labels[:, 0] == 2)][:, 1])
        pids_LBBB = list(labels[np.where(labels[:, 0] == 3)][:, 1])
        pids_RBBB = list(labels[np.where(labels[:, 0] == 4)][:, 1])
        pids_PAC = list(labels[np.where(labels[:, 0] == 5)][:, 1])
        pids_PVC = list(labels[np.where(labels[:, 0] == 6)][:, 1])
        pids_STD = list(labels[np.where(labels[:, 0] == 7)][:, 1])
        pids_STE = list(labels[np.where(labels[:, 0] == 8)][:, 1])

        print(f"=> Dataset {name} has {len(labels)} patients, with {len(pids_Normal)} Normal, {len(pids_AF)} AF, {len(pids_IAVB)} IAVB, {len(pids_LBBB)} LBBB, {len(pids_RBBB)} RBBB, {len(pids_PAC)} PAC, {len(pids_PVC)} PVC, {len(pids_STD)} STD, {len(pids_STE)} STE")
        
        idxs = np.rint([0.1*len(pids_Normal), 0.1*len(pids_AF), 0.1*len(pids_IAVB), 0.1*len(pids_LBBB), 0.1*len(pids_RBBB), 0.1*len(pids_PAC), 0.1*len(pids_PVC), 0.1*len(pids_STD), 0.1*len(pids_STE)]).astype(int)
        train_ids = pids_Normal[:-2*idxs[0]] + pids_AF[:-2*idxs[1]] + pids_IAVB[:-2*idxs[2]] + pids_LBBB[:-2*idxs[3]] + pids_RBBB[:-2*idxs[4]] + pids_PAC[:-2*idxs[5]] + pids_PVC[:-2*idxs[6]] + pids_STD[:-2*idxs[7]] + pids_STE[:-2*idxs[8]]
        val_ids = pids_Normal[-2*idxs[0]:-idxs[0]] + pids_AF[-2*idxs[1]:-idxs[1]] + pids_IAVB[-2*idxs[2]:-idxs[2]] + pids_LBBB[-2*idxs[3]:-idxs[3]] + pids_RBBB[-2*idxs[4]:-idxs[4]] + pids_PAC[-2*idxs[5]:-idxs[5]] + pids_PVC[-2*idxs[6]:-idxs[6]] + pids_STD[-2*idxs[7]:-idxs[7]] + pids_STE[-2*idxs[8]:-idxs[8]]
        test_ids = pids_Normal[-idxs[0]:] + pids_AF[-idxs[1]:] + pids_IAVB[-idxs[2]:] + pids_LBBB[-idxs[3]:] + pids_RBBB[-idxs[4]:] + pids_PAC[-idxs[5]:] + pids_PVC[-idxs[6]:] + pids_STD[-idxs[7]:] + pids_STE[-idxs[8]:]
        
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    print(f"=> Split the dataset into {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
    
    return labels, train_ids, val_ids, test_ids


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    see https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=0)
    return y


def process_ts(ts, fs, normalized=True, bandpass_filter=False):
    """ preprocess a time-series data

    Args:
        ts (numpy.ndarray): The input time-series in shape (timestamps, feature).
        fs (float): The sampling frequency for bandpass filtering.
        normalized (bool): Whether to normalize the time-series data.
        bandpass_filter (bool): Whether to filter the time-series data.

    Returns:
        ts (numpy.ndarray): The processed time-series.
    """
    if bandpass_filter:
        ts = butter_bandpass_filter(ts, 0.5, 50, fs, 5)
    if normalized:
        scaler = StandardScaler()
        scaler.fit(ts)
        ts = scaler.transform(ts)
    return ts


def process_batch_ts(batch, fs=256, normalized=True, bandpass_filter=False):
    """ preprocess a batch of time-series data

    Args:
        batch (numpy.ndarray): A batch of input time-series in shape (n_samples, timestamps, feature).

    Returns:
        A batch of processed time-series.
    """
    bool_iterator_1 = repeat(fs, len(batch))
    bool_iterator_2 = repeat(normalized, len(batch))
    bool_iterator_3 = repeat(bandpass_filter, len(batch))
    return np.array(list(map(process_ts, batch, bool_iterator_1, bool_iterator_2, bool_iterator_3)))


def split_data_label(X_trial, y_trial, sample_timestamps, overlapping, keep_dim=False):
    """ split a batch of time-series trials into samples and adding trial ids to the label array y

    Args:
        X_trial (numpy.ndarray): It should have a shape of (n_trials, trial_timestamps, features) B_trial x T_trial x C.
        y_trial (numpy.ndarray): It should have a shape of (n_trials, 2). The first column is the label and the second column is patient id.
        sample_timestamps (int): The length for sample-level data (T_sample).
        overlapping (float): How many overlapping for each sample-level data in a trial.

    Returns:
        X_sample (numpy.ndarray): It should have a shape of (n_samples, sample_timestamps, features) B_sample x T_sample x C. The B_sample = B x sample_num.
        y_sample (numpy.ndarray): It should have a shape of (n_samples, 3). The three columns are the label, patient id, and trial id.
    """
    X_sample, trial_ids, sample_num = split_data(X_trial, sample_timestamps, overlapping, keep_dim=keep_dim)
    # all samples from same trial should have same label and patient id
    if keep_dim:
        y_sample = y_trial
    else:
        y_sample = np.repeat(y_trial, repeats=sample_num, axis=0)
    # append trial ids. Segments split from same trial should have same trial ids
    label_num = y_sample.shape[0]
    y_sample = np.hstack((y_sample.reshape((label_num, -1)), trial_ids.reshape((label_num, -1))))
    return X_sample, y_sample


def split_data(X_trial, sample_timestamps=256, overlapping=0.5, keep_dim=False):
    """ split a batch of trials into samples and mark their trial ids

    Args:
        See split_data_label() function

    Returns:
        X_sample (numpy.ndarray): (n_samples, sample_timestamps, feature).
        trial_ids (numpy.ndarray): (n_samples,)
        sample_num (int): one trial splits into sample_num of samples
    """
    
    length = X_trial.shape[1]
    # check if sub_length and overlapping compatible
    if overlapping:
        assert (length - (1-overlapping)*sample_timestamps) % (sample_timestamps*overlapping) == 0
        sample_num = (length - (1 - overlapping) * sample_timestamps) / (sample_timestamps * overlapping)
    else:
        assert length % sample_timestamps == 0
        sample_num = length / sample_timestamps
    sample_feature_list = []
    trial_id_list = []
    trial_id = 1
    for trial in X_trial:
        if keep_dim:
            sample_feature = []
            
        counter = 0
        while counter*sample_timestamps*(1-overlapping)+sample_timestamps <= trial.shape[0]:
            if keep_dim:
                sample_feature.append(
                    trial[int(counter*sample_timestamps*(1-overlapping)):int(counter*sample_timestamps*(1-overlapping)+sample_timestamps)]
                    )
            else:
                sample_feature_list.append(
                    trial[int(counter*sample_timestamps*(1-overlapping)):int(counter*sample_timestamps*(1-overlapping)+sample_timestamps)]
                    )
                trial_id_list.append(trial_id)
            counter += 1
            
        if keep_dim:
            sample_feature = np.array(sample_feature)
            sample_feature_list.append(sample_feature)
            trial_id_list.append(trial_id)
            
        trial_id += 1
        
    X_sample, trial_ids = np.array(sample_feature_list), np.array(trial_id_list)

    return X_sample, trial_ids, sample_num
        
