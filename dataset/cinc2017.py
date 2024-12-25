import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from wfdb import rdrecord
from tqdm import tqdm


class CINC2017Dataset(Dataset):
    '''CINC2017 dataset
    
    Args:
        root (str): root directory of the dataset
        length (int): length of the segmented signal, if None, use the whole signal
        step (int): step of the sliding window, if None, set to length
        split (str): one of 'train', 'valid' or 'test'
        transform (transform): data augmentation
    '''
    def __init__(self, root='training2017', length=None, step=None, split='train', drop_noise=True, transform=None):
        self.transform = transform
        self.length = length
        self.step = step
        self.split = split
        self.root = root
        self.drop_noise = drop_noise
        self.classes = ['N', 'A', 'O', '~']
        self.data = pd.merge(self._load_label(), self._load_data(), on='head',how='inner')
        
        # don't use the noise class
        if self.drop_noise:
            self.data = self.data[self.data['label'] != 3]
            
        if self.length:
            self._segment_data()

        
    def __len__(self):
        return len(self.data)
        
        
    def __getitem__(self, idx):
        '''
        Returns:
            signal (np.array): with shape (lead, length) where lead is 1 here.
            label (int): label of the signal
        '''    
        signal = self.data.at[idx, 'signal']
        label = self.data.at[idx, 'label']
        if self.transform:
            signal = self.transform(signal)
        return signal, label
    
    
    def _load_label(self):
        '''
        read the label file and return the label dataframe
        '''
        labels = pd.read_csv(os.path.join(self.root, 'REFERENCE-v3.csv'), header=None)
        labels.columns = ['head', 'label']
        label_dict = {'N': 0, 'A': 1, 'O': 2, '~': 3}
        labels.replace(label_dict, inplace=True)
        return labels


    def _load_data(self):
        '''
        read corresponding split dataframe
        '''
        # read all signals' name
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
            signals = pd.concat([signals,
                                 pd.DataFrame({'head': h, 'signal': [signal]})])
        signals.reset_index(drop=True, inplace=True)
            
        return signals


    def _segment_data(self):
        '''
        perform segmentation on the signals
        '''
        # for balancing the label, specify the stride of the sliding window of each class
        # if self.step == None or self.step == 0 or self.step == 'none':
        #     inter_n = inter_a = inter_o = inter_p = self.length
        # elif isinstance(self.step, dict):
        #     inter_n, inter_a, inter_o, inter_p = self.step.values()
        # elif isinstance(self.step, (int, float)):
        #     inter_n = inter_a = inter_o = inter_p = self.step
        # elif isinstance(self.step, (list, tuple)):
        #     inter_n, inter_a, inter_o, inter_p = self.step
        # else:
        #     raise ValueError('step must be dict/list/tuple/number')
        if self.step is None or self.step == 0 or self.step == 'none':
            self.step = self.length
        if isinstance(self.step, float):
            self.step = int(self.length * self.step)
        else:
            raise ValueError('step must be a number')
        
        # segment the signal
        def slicing_window(signal, window, interval):
            return [signal[..., i:i+window] for i in range(0, signal.shape[-1], interval)
                    if i+window <= signal.shape[-1]]
            
            
        for row in self.data.itertuples():
            if row.signal.shape[-1] >= self.length:
                self.data.at[row.Index, 'signal'] = slicing_window(row.signal, self.length, self.step)
                # if row.label == 0:
                #     self.data.at[row.Index, 'signal'] =\
                #         slicing_window(row.signal, self.length, inter_n)
                # elif row.label == 1:
                #     self.data.at[row.Index, 'signal'] =\
                #         slicing_window(row.signal, self.length, inter_a)
                # elif row.label == 2:
                #     self.data.at[row.Index, 'signal'] =\
                #         slicing_window(row.signal, self.length, inter_o)
                # else:
                #     self.data.at[row.Index, 'signal'] =\
                #         slicing_window(row.signal, self.length, inter_p)   
            else:
                self.data.drop(row.Index, inplace=True)
                
        self.data = self.data.explode('signal', ignore_index=True)
        

