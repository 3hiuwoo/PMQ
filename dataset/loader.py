import random
import torch
from torch.utils.data import DataLoader, TensorDataset, BatchSampler
from base import load_chapman
from utils.functional import trial_shuffle

def load_data(root, task, batch_size=256, dataset_name='chapman'):
    '''
    return dataloaders based on the task and dataset
    '''
    if task in ['comet', 'isl']:
        if dataset_name == 'chapman':
            X_train, _, _, y_train, _, _ = load_chapman(root=root, split=True)
            X_train, y_train = trial_shuffle(X_train, y_train)
            
            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
            
            sampler = SSBatchSampler(range(len(train_dataset)), batch_size, drop_last=True)
            train_loader = DataLoader(train_dataset, batch_sampler=sampler)
            
            return train_loader, 12 # in_channels
            
        else:
            raise ValueError('Dataset not supported')
        
    elif task in ['cmsc']:
        if dataset_name == 'chapman':
            X_train, _, _, y_train, _, _ = load_chapman(root=root, split=True)
            X_train, y_train = trial_shuffle(X_train, y_train)
            
            X_train, y_train = X_train.reshape(-1, 2, X_train.shape[1], X_train.shape[2]), y_train.reshape(-1, 2, y_train.shape[1])
            
            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
            
            sampler = SSBatchSampler(range(len(train_dataset)), batch_size, drop_last=True)
            train_loader = DataLoader(train_dataset, batch_sampler=sampler)
            
            return train_loader, 12 # in_channels
             
    elif task == 'supervsied':
        if dataset_name == 'chapman':
            X_train, X_valid, X_test, y_train, y_valid, y_test = load_chapman(root=root, split=True)
            
            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
            valid_dataset = TensorDataset(torch.tensor(X_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.long))
            test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            return train_loader, valid_loader, test_loader, 12 # in_channels
        
        else:
            raise ValueError('Dataset not supported') 
    else:
        raise ValueError('Task not supported')
    
    
class SSBatchSampler(BatchSampler):
    '''
    batch sampler for COMET/CLOCS, sequentially sample the batch and shuffle in-batch position,
    the argument sampler should be a sequential iterator
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