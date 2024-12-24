from torch.utils.data import DataLoader
from .cinc2017 import CINC2017Dataset
from .chapman import ChapmanDataset
from utils.transform import Compose, Normalize, ToTensor

def load_data(root, task, transform=None, batch_size=256, dataset_name='cinc2017'):
    '''
    return dataloaders
    '''
    if task in ['cmsc', 'simclr', 'moco', 'mcp']:
        if dataset_name == 'cinc2017':
            train_dataset = CINC2017Dataset(root=root, length=5000, split='train', transform=transform)
        
        elif dataset_name == 'chapman':
            train_dataset = ChapmanDataset(root=root, split='train', keep_lead=False, transform=transform)

        else:
            raise ValueError(f'Unknown dataset {dataset_name}')
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        return train_dataloader
          
    elif task == 'supervised':
        if dataset_name == 'cinc2017':
            train_dataset = CINC2017Dataset(root=root, length=2500, split='train', transform=transform)
            valid_dataset = CINC2017Dataset(root=root, length=2500, split='valid', transform=ToTensor())
            test_dataset = CINC2017Dataset(root=root, length=2500, split='test', transform=ToTensor())
            
        elif dataset_name == 'chapman':
            train_dataset = ChapmanDataset(root=root, split='train', pretrain=False, transform=transform)
            valid_dataset = ChapmanDataset(root=root, split='valid', pretrain=False, transform=Compose([Normalize(), ToTensor()]))
            test_dataset = ChapmanDataset(root=root, split='test', pretrain=False, transform=Compose([Normalize(), ToTensor()]))
            
        else:
            raise ValueError(f'Unknown dataset {dataset_name}')
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
        return train_dataloader, valid_dataloader, test_dataloader 
    
    else:
        raise ValueError(f'Unknown task {task}')
    
    
