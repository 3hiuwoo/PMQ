import random
from torch.utils.data import DataLoader, BatchSampler
from .cinc2017 import CINC2017Dataset
from .chapman import ChapmanDataset
from utils.transform import Compose, Normalize, ToTensor

def load_data(root, task, transform=None, batch_size=256, dataset_name='cinc2017'):
    '''
    return dataloaders based on the task and dataset
    '''
    if task in ['cmsc', 'simclr', 'moco', 'mcp']:
        if dataset_name == 'chapman':
            train_dataset = ChapmanDataset(root=root, split='train', keep_lead=False, transform=transform)

        elif  dataset_name == 'chapman_lead': # don't flatten the lead dimension
            train_dataset = ChapmanDataset(root=root, split='train', transform=transform)

        else:
            raise ValueError(f'Unknown dataset {dataset_name}')
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        return train_dataloader
    
    elif task in ['comet']:
        if dataset_name == 'chapman_trial':
            train_dataset = ChapmanDataset(root=root, split='train', trial=2, sample=250, transform=transform)
            
        else:
            raise ValueError(f'Unknown dataset {dataset_name}')
        
        sampler = CometBatchSampler(range(len(train_dataset)), batch_size, drop_last=True)
        train_dataloader = DataLoader(train_dataset, batch_sampler=sampler)\
        # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
             
    elif task == 'supervised':
        if dataset_name == 'cinc2017':
            train_dataset = CINC2017Dataset(root=root, length=2500, split='train', transform=transform)
            valid_dataset = CINC2017Dataset(root=root, length=2500, split='valid', transform=ToTensor())
            test_dataset = CINC2017Dataset(root=root, length=2500, split='test', transform=ToTensor())
            
        elif dataset_name == 'chapman':
            train_dataset = ChapmanDataset(root=root, split='train', pretrain=False, transform=transform)
            valid_dataset = ChapmanDataset(root=root, split='valid', pretrain=False, transform=Compose([Normalize(), ToTensor()]))
            test_dataset = ChapmanDataset(root=root, split='test', pretrain=False, transform=Compose([Normalize(), ToTensor()]))
        
        elif dataset_name == 'chapman_lead':
            train_dataset = ChapmanDataset(root=root, split='train', pretrain=False, keep_lead=False, transform=transform)
            valid_dataset = ChapmanDataset(root=root, split='valid', pretrain=False, keep_lead=False, transform=Compose([Normalize(), ToTensor()]))
            test_dataset = ChapmanDataset(root=root, split='test', pretrain=False, keep_lead=False, transform=Compose([Normalize(), ToTensor()]))
            
        elif dataset_name == 'chapman_trial':
            train_dataset = ChapmanDataset(root=root, split='train', pretrain=False, trial=2, sample=250, transform=transform)
            valid_dataset = ChapmanDataset(root=root, split='valid', pretrain=False, trial=2, sample=250, transform=Compose([Normalize(), ToTensor()]))
            test_dataset = ChapmanDataset(root=root, split='test', pretrain=False, trial=2, sample=250, transform=Compose([Normalize(), ToTensor()]))
                
        else:
            raise ValueError(f'Unknown dataset {dataset_name}')
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
        return train_dataloader, valid_dataloader, test_dataloader 
    
    else:
        raise ValueError(f'Unknown task {task}')
    
    
class CometBatchSampler(BatchSampler):
    '''
    batch sampler for COMET, the argument sampler should be a sequential iterator
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