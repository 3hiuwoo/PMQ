from torch.utils.data import DataLoader
from .cinc2017 import CINC2017Dataset
from .chapman import ChapmanDataset
from utils.transform import Compose, Normalize, ToTensor

def load_data(root, task, transform=None, batch_size=256, dataset_name='cinc2017'):
    '''
    return dataloaders
    '''
    if task == 'contrast':
        if dataset_name == 'cinc2017':
            train_dataset = CINC2017Dataset(root=root, length=5000, split='train', transform=transform)
            valid_dataset = CINC2017Dataset(root=root, length=5000, split='valid', transform=ToTensor())
            test_dataset = CINC2017Dataset(root=root, length=5000, split='test', transform=ToTensor())
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        elif dataset_name == 'chapman':
            train_dataset = ChapmanDataset(root=root, split='train', transform=transform)
            valid_dataset = ChapmanDataset(root=root, split='valid', transform=Compose([Normalize(), ToTensor()]))
            test_dataset = ChapmanDataset(root=root, split='test', transform=Compose([Normalize(), ToTensor()]))
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        else:
            raise ValueError(f'Unknown dataset {dataset_name}')
        
    elif task == 'supervised':
        if dataset_name == 'cinc2017':
            train_dataset = CINC2017Dataset(root=root, length=2500, split='train', transform=transform)
            valid_dataset = CINC2017Dataset(root=root, length=2500, split='valid', transform=ToTensor())
            test_dataset = CINC2017Dataset(root=root, length=2500, split='test', transform=ToTensor())
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        elif dataset_name == 'chapman':
            # downsample class of transformation is needed and not yet implemented
            train_dataset = ChapmanDataset(root=root, split='train', transform=transform)
            valid_dataset = ChapmanDataset(root=root, split='valid', transform=Compose([Normalize(), ToTensor()]))
            test_dataset = ChapmanDataset(root=root, split='test', transform=Compose([Normalize(), ToTensor()]))
            
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        else:
            raise ValueError(f'Unknown dataset {dataset_name}')
          
    else:
        raise ValueError(f'Unknown task {task}')
    
    return train_dataloader, valid_dataloader, test_dataloader
