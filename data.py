import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter
from itertools import repeat
    
def load_data(root='dataset', name='chapman', length=None, overlap=0, norm=True):
    ''' load and preprocess data
    '''
    data_path = os.path.join(root, name, 'feature')
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
    
    if name == 'ptb':
        X_train = X_train[:, :, :12]
        X_val = X_val[:, :, :12]
        X_test = X_test[:, :, :12]

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


def load_split_ids(root='dataset', name='chapman'):
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
        
        train_ids = pids_norm[:-1200] + pids_mi[:-600] + pids_sttc[:-600] + pids_cd[:-400] + pids_hyp[:-200]
        val_ids = pids_norm[-1200:-600] + pids_mi[-600:-300] + pids_sttc[-600:-300] + pids_cd[-400:-200] + pids_hyp[-200:-100]
        test_ids = pids_norm[-600:] + pids_mi[-300:] + pids_sttc[-300:] + pids_cd[-200:] + pids_hyp[-100:]
    
    elif name == 'cpsc2018':
        labels[:, 0] -= 1
        
        pids_Normal = list(labels[np.where(labels[:, 0] == 0)][:, 1])
        pids_AF = list(labels[np.where(labels[:, 0] == 1)][:, 1])
        pids_IAVB = list(labels[np.where(labels[:, 0] == 2)][:, 1])
        pids_LBBB = list(labels[np.where(labels[:, 0] == 3)][:, 1])
        pids_RBBB = list(labels[np.where(labels[:, 0] == 4)][:, 1])
        pids_PAC = list(labels[np.where(labels[:, 0] == 5)][:, 1])
        pids_PVC = list(labels[np.where(labels[:, 0] == 6)][:, 1])
        pids_STD = list(labels[np.where(labels[:, 0] == 7)][:, 1])
        pids_STE = list(labels[np.where(labels[:, 0] == 8)][:, 1])

        train_ids = pids_Normal[:-300] + pids_AF[:-300] + pids_IAVB[:-300] + pids_LBBB[:-100] + pids_RBBB[:-300] + pids_PAC[:-300] + pids_PVC[:-300] + pids_STD[:-300] + pids_STE[:-100]
        val_ids = pids_Normal[-300:-150] + pids_AF[-300:-150] + pids_IAVB[-300:-150] + pids_LBBB[-100:-50] + pids_RBBB[-300:-150] + pids_PAC[-300:-150] + pids_PVC[-300:-150] + pids_STD[-300:-150] + pids_STE[-100:-50]
        test_ids = pids_Normal[-150:] + pids_AF[-150:] + pids_IAVB[-150:] + pids_LBBB[-50:] + pids_RBBB[-150:] + pids_PAC[-150:] + pids_PVC[-150:] + pids_STD[-150:] + pids_STE[-50:]
    
    elif name == 'cinc2017':
        pids_n = list(labels[np.where(labels[:, 0]==0)][:, 1])
        pids_a = list(labels[np.where(labels[:, 0]==1)][:, 1])
        pids_o = list(labels[np.where(labels[:, 0]==2)][:, 1])
        
        # val 50 50 50 test 300 300 300
        train_ids = pids_n[:-900] + pids_a[:-300] + pids_o[:-800]
        val_ids = pids_n[-900:-450] + pids_a[-300:-150] + pids_o[-800:-400]
        test_ids = pids_n[-450:] + pids_a[-150:] + pids_o[-400:]
        
    else:
        raise ValueError(f'Unknown dataset: {name}')
        
    return labels, train_ids, val_ids, test_ids


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    ''' see https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter

    '''
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=0)
    return y


def process_ts(ts, fs, normalized=True, bandpass_filter=False):
    ''' preprocess a time-series data

    Args:
        ts (numpy.ndarray): The input time-series in shape (timestamps, feature).
        fs (float): The sampling frequency for bandpass filtering.
        normalized (bool): Whether to normalize the time-series data.
        bandpass_filter (bool): Whether to filter the time-series data.

    Returns:
        ts (numpy.ndarray): The processed time-series.
    '''

    if bandpass_filter:
        ts = butter_bandpass_filter(ts, 0.5, 50, fs, 5)
    if normalized:
        scaler = StandardScaler()
        scaler.fit(ts)
        ts = scaler.transform(ts)
    return ts


def process_batch_ts(batch, fs=256, normalized=True, bandpass_filter=False):
    ''' preprocess a batch of time-series data

    Args:
        batch (numpy.ndarray): A batch of input time-series in shape (n_samples, timestamps, feature).

    Returns:
        A batch of processed time-series.
    '''

    bool_iterator_1 = repeat(fs, len(batch))
    bool_iterator_2 = repeat(normalized, len(batch))
    bool_iterator_3 = repeat(bandpass_filter, len(batch))
    return np.array(list(map(process_ts, batch, bool_iterator_1, bool_iterator_2, bool_iterator_3)))


def split_data_label(X_trial, y_trial, sample_timestamps, overlapping):
    ''' split a batch of time-series trials into samples and adding trial ids to the label array y

    Args:
        X_trial (numpy.ndarray): It should have a shape of (n_trials, trial_timestamps, features) B_trial x T_trial x C.
        y_trial (numpy.ndarray): It should have a shape of (n_trials, 2). The first column is the label and the second column is patient id.
        sample_timestamps (int): The length for sample-level data (T_sample).
        overlapping (float): How many overlapping for each sample-level data in a trial.

    Returns:
        X_sample (numpy.ndarray): It should have a shape of (n_samples, sample_timestamps, features) B_sample x T_sample x C. The B_sample = B x sample_num.
        y_sample (numpy.ndarray): It should have a shape of (n_samples, 3). The three columns are the label, patient id, and trial id.
    '''
    X_sample, trial_ids, sample_num = split_data(X_trial, sample_timestamps, overlapping)
    # all samples from same trial should have same label and patient id
    y_sample = np.repeat(y_trial, repeats=sample_num, axis=0)
    # append trial ids. Segments split from same trial should have same trial ids
    label_num = y_sample.shape[0]
    y_sample = np.hstack((y_sample.reshape((label_num, -1)), trial_ids.reshape((label_num, -1))))
    # X_sample, y_sample = shuffle(X_sample, y_sample)
    return X_sample, y_sample


def split_data(X_trial, sample_timestamps=256, overlapping=0.5):
    ''' split a batch of trials into samples and mark their trial ids

    Args:
        See split_data_label() function

    Returns:
        X_sample (numpy.ndarray): (n_samples, sample_timestamps, feature).
        trial_ids (numpy.ndarray): (n_samples,)
        sample_num (int): one trial splits into sample_num of samples
    '''
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
        counter = 0
        # ex. split one trial(5s, 1280 timestamps) into 9 half-overlapping samples (1s, 256 timestamps)
        while counter*sample_timestamps*(1-overlapping)+sample_timestamps <= trial.shape[0]:
            sample_feature = trial[int(counter*sample_timestamps*(1-overlapping)):int(counter*sample_timestamps*(1-overlapping)+sample_timestamps)]
            # print(f"{int(counter*length*(1-overlapping))}:{int(counter*length*(1-overlapping)+length)}")
            sample_feature_list.append(sample_feature)
            trial_id_list.append(trial_id)
            counter += 1
        trial_id += 1
    X_sample, trial_ids = np.array(sample_feature_list), np.array(trial_id_list)

    return X_sample, trial_ids, sample_num
        
