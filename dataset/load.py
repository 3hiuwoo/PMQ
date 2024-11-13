from torch.utils.data import DataLoader
from .cinc2017 import CINC2017Dataset
from .chapman import ChapmanDataset
from utils.transform import Compose, Normalize, ToTensor

def load_data(root, transform=None, batch_size=256, dataset_name='cinc2017', *, length=2500, step=None, leads=None, pretrain=True):
    if dataset_name == 'cinc2017':
        train_dataset = CINC2017Dataset(root=root, length=length, step=step, split='train', transform=transform)
        valid_dataset = CINC2017Dataset(root=root, length=length, step=step, split='valid', transform=ToTensor())
        test_dataset = CINC2017Dataset(root=root, length=length, step=step, split='test', transform=ToTensor())
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if dataset_name == 'chapman':
        if not pretrain:
            print('=> using chapman dataset for supervised training, should notice that there are 43044/45152 ecg signals available')

        train_dataset = ChapmanDataset(root=root, split='train', pretrain=pretrain, leads=leads, transform=transform)
        valid_dataset = ChapmanDataset(root=root, split='valid', pretrain=pretrain, leads=leads, transform=Compose([Normalize(), ToTensor()]))
        test_dataset = ChapmanDataset(root=root, split='test', pretrain=pretrain, leads=leads, transform=Compose([Normalize(), ToTensor()]))
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
    return train_dataloader, valid_dataloader, test_dataloader
