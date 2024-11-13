# -*- coding: utf-8 -*-
import torch.nn as nn
import torch

class CNN3(nn.Module):
    def __init__(self, embeddim=256, kernel_sizes=7, kernel_strides=3, channels=(4, 16, 32), dropouts=0.1):
        super(CNN3, self).__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts]*3
        if isinstance(channels, int):
            channels = [channels]*3
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]*3
        if isinstance(kernel_strides, int):
            kernel_strides = [kernel_strides]*3
        self.backbone = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_sizes[0], kernel_strides[0]),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropouts[0]),
            
            nn.Conv1d(channels[0], channels[1], kernel_sizes[1], kernel_strides[1]),
            nn.BatchNorm1d(channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropouts[1]),
            
            nn.Conv1d(channels[1], channels[2], kernel_sizes[2], kernel_strides[2]),
            nn.BatchNorm1d(channels[2]),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropouts[2]),
            
            nn.Flatten(),
            nn.Linear(10 * channels[2], embeddim)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        return x
    
    
class CNNContrast(nn.Module):
    def __init__(self, network=CNN3, embeddim=256, kernel_sizes=7, kernel_strides=3,
                 channels=(4, 16, 32), dropouts=0.1):
        super(CNNContrast,self).__init__()
        self.embeddim = embeddim
        self.encoder = network(embeddim, kernel_sizes, kernel_strides, channels, dropouts)
        
        
    def forward(self,x):
        """
        Args:
            x (torch.Tensor): inputs with N views (BxNxCxS)
        Returns:
            h (torch.Tensor): latent embedding for each of the N views (NxBxH)
        """
        batch_size = x.shape[0]
        nviews = x.shape[1]
        x = x.permute(1, 0, 2, 3)
        embeddings = torch.empty(nviews, batch_size, self.embeddim)
        for n in range(nviews):       
            """ Obtain Inputs From Each View """
            h = x[n, ...]
            h = self.encoder(h)
            embeddings[n, ...] = h
        return embeddings


class CNNSupervised(nn.Module):
    def __init__(self, network=CNN3, num_classes=4, embeddim=256, kernel_sizes=7, kernel_strides=3,
                 channels=(4, 16, 32), dropouts=0.1):
        super(CNNSupervised,self).__init__()
        self.embeddim = embeddim
        self.encoder = network(embeddim, kernel_sizes, kernel_strides, channels, dropouts)
        self.fc = nn.Linear(embeddim, num_classes)
        
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x