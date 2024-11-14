import torch
import numpy as np
import random
import os
import shutil


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

def get_device():
    return (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.mps.is_available()
        else 'cpu'
        )
    
    
def save_checkpoint(checkpoint, is_best, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
    torch.save(checkpoint, path)
    if is_best:
        shutil.copyfile(path, os.path.join(os.path.dirname(path), 'model_best.pth'))
    