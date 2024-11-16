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
        transform (transform): data augmentation
    '''
    def __init__(self, root='trainingchapman', split='train', pretrain=True, keep_lead=None, transform=None):
        self.transform = transform
        self.split = split
        self.root = root
        self.keep_lead = keep_lead
        self.pretrain = pretrain
        self.classes = ['SB', 'AFIB', 'GSVT', 'SR']
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.data = self._load_data()
        if not self.pretrain:
            self.data = pd.merge(self._load_label(), self.data, on='head', how='inner')

        
    def __len__(self):
        return len(self.data)
        
        
    def __getitem__(self, idx):
        signal = self.data.at[idx, 'signal'][np.newaxis, :]
        if self.pretrain:
            head = self.data.at[idx, 'head']
            if self.transform:
                signal = self.transform(signal)
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
        read corresponding subset dataframe
        '''
        # read all signals' name in the corresponding dataset
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
            signal = rdrecord(os.path.join(self.root, h)).p_signal.T
            
            # drop leads that contain NaN
            mask = np.all(~np.isnan(signal), axis=1)
            signal = signal[mask]
            
            # drop leads that return empty signal
            mask = np.any(signal!=0, axis=1)
            signal = signal[mask]
            
            signals = pd.concat([signals,
                                 pd.DataFrame({'head': h, 'signal': [signal]})])
            
        signals.reset_index(drop=True, inplace=True)
        
        if self.keep_lead is None:
            signals = signals.explode('signal', ignore_index=True)
        # else: ...
        
        return signals

        

