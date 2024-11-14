import torch
from torch import nn
from model.basic import CNN3


def load_model(model_name, task, embeddim):
    if task == 'contrast':
        if model_name == 'cnn3':
            return ContrastModel(network=CNN3, embeddim=embeddim)
    elif task == 'supervised':
        if model_name == 'cnn3':
            return SupervisedModel(network=CNN3, embeddim=embeddim)
        

class ContrastModel(nn.Module):
    def __init__(self, network, embeddim=256, kernel_sizes=7, kernel_strides=3,
                 channels=(4, 16, 32), dropouts=0.1):
        super(ContrastModel,self).__init__()
        self.embeddim = embeddim
        self.encoder = network(embeddim, kernel_sizes, kernel_strides, channels, dropouts)
        
        
    def forward(self,x):
        """
        Args:
            x (torch.Tensor): inputs with N views (BxNxCxS)
        Returns:
            h (torch.Tensor): latent embedding for each of the N views (NxBxH)
        """
        nviews = x.shape[1]
        x = x.permute(1, 0, 2, 3)
        h = [self.encoder(x[n, ...]) for n in range(nviews)]
        return torch.stack(h, dim=0)
    

class SupervisedModel(nn.Module):
    def __init__(self, network, num_classes=4, embeddim=256, kernel_sizes=7, kernel_strides=3,
                 channels=(4, 16, 32), dropouts=0.1):
        super(SupervisedModel,self).__init__()
        self.embeddim = embeddim
        self.encoder = network(embeddim, kernel_sizes, kernel_strides, channels, dropouts)
        self.fc = nn.Linear(embeddim, num_classes)
        
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x