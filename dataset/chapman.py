import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from wfdb import rdrecord
from tqdm import tqdm


class ChapmanDataset(Dataset):
    '''Chapman dataset
    
    Args:
        root (str): root directory of the dataset
        split (str): one of 'train', 'valid' or 'test'
        pretrain (bool): if True, return head, if False, return label
        keep_lead (bool): whether to keep the dimension of lead
        trial(int): the trial number of each signal to be segmented, if None/0/'none', use the whole signal
        sample(int): the sample length of each segment extracted from the trial, if None/0/'none', use the whole trial
        transform (transform): data augmentation
    '''
    def __init__(self, root='trainingchapman', split='train', pretrain=True, keep_lead=True, trial=None, sample=None, transform=None):
        self.transform = transform
        self.split = split
        self.root = root
        self.keep_lead = keep_lead
        self.pretrain = pretrain
        self.trial = trial
        self.sample = sample
        self.classes = ['SB', 'AFIB', 'GSVT', 'SR']
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.data = self._load_data()
        if not self.pretrain:
            self.data = pd.merge(self._load_label(), self.data, on='head', how='inner')
            
        if self.trial:
            self._make_trial()
        if self.sample:
            self._make_sample()

        
    def __len__(self):
        return len(self.data)
        
        
    def __getitem__(self, idx):
        '''
        Returns:
            signal (np.array): with shape (lead, length) where lead is 1 here.
            head(str)/label(int): if pretrain is True, return head, else return label
        '''
        signal = self.data.at[idx, 'signal']
        # ensure the signal has 2 dimensions corresponding to (lead, length)
        signal = signal.reshape(-1, signal.shape[-1])
        if self.pretrain:
            head = self.data.at[idx, 'head']
            if self.transform:
                signal = self.transform(signal)
            if self.trial:
                trial = self.data.at[idx, 'trial']
                return signal, head, trial
            else:
                return signal, head
            
        else:
            label = self.data.at[idx, 'label']
            if self.transform:
                signal = self.transform(signal)
            return signal, label
 
    
    def _load_label(self):
        labels = pd.read_csv(os.path.join(self.root, 'REFERENCE4.csv'))
        label_dict = {'SB': 0, 'AFIB': 1, 'GSVT': 2, 'SR': 3}
        labels.replace(label_dict, inplace=True)
        return labels


    def _load_data(self):
        '''
        read corresponding split dataframe
        '''
        # read all signals' directory
        if self.split == 'train':
            with open(os.path.join(self.root, 'RECORDTrain.txt'), 'r') as f:
                heads = f.readlines()
            heads = [h.strip() for h in heads]
        elif self.split == 'valid':
            with open(os.path.join(self.root, 'RECORDValid.txt'), 'r') as f:
                heads = f.readlines()
            heads = [h.strip() for h in heads]
        elif self.split == 'test':
            with open(os.path.join(self.root, 'RECORDTest.txt'), 'r') as f:
                heads = f.readlines()
            heads = [h.strip() for h in heads]
        else:
            raise ValueError('split must be one of "train", "valid" or "test"')

        # use wfdb to read .mat file and append the signal to the dataframe
        signals = pd.DataFrame(columns=['head', 'signal'])
        for h in tqdm(heads, desc=f'=> Loading {self.split} dataset'):
            
            # read all signals' name in the directory
            with open(os.path.join(self.root, h, 'RECORDS'), 'r') as f:
                names = f.readlines()
            names = [n.strip() for n in names]
            
            for n in names:
                signal = rdrecord(os.path.join(self.root, h, n)).p_signal.T
                
                # drop signals that contain NaN
                if np.isnan(signal).any():
                    continue
            
                # drop signals that have leads with all zeros
                if np.all(signal == 0, axis=1).any():
                    continue
            
                signals = pd.concat([signals,
                                    pd.DataFrame({'head': n, 'signal': [signal]})])
           
        signals.reset_index(drop=True, inplace=True)
        
        # flatten the signal by lead dimension
        if not self.keep_lead:
            signals = signals.explode('signal', ignore_index=True)
        
        return signals


    def _make_trial(self):
        '''
        segment each signal into trials and add trial id
        '''
        length = self.data['signal'].apply(lambda x: x.shape[-1]).min()
        assert length == 5000
        T = length // self.trial
        self.data['signal'] = self.data['signal'].apply(lambda x: self._segment(x, T))
        tid = np.tile(np.arange(self.trial), self.data.shape[0])
        self.data = self.data.explode('signal', ignore_index=True)
        self.data['trial'] = tid
        
    
    def _make_sample(self):
        '''
        segment each trial into samples
        '''
        self.data['signal'] = self.data['signal'].apply(lambda x: self._segment(x, self.sample))
        self.data = self.data.explode('signal', ignore_index=True)


    def _segment(self, signal, length):
        '''
        segment the signal non-overlapping
        '''
        return [signal[..., i:i+length] for i in range(0, signal.shape[-1], length)
                    if i+length <= signal.shape[-1]]