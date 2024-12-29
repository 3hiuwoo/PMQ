import torch
from torch import nn
from torch.nn import functional as F

class CNN3(nn.Module):
    '''
    a convolutional neural network with 3 convolutional layers
    '''
    def __init__(self, in_channels=1, embeddim=256, keep_dim=False):
        super(CNN3, self).__init__()
        
        kernel_sizes = [7] * 3
        kernel_strides = [3] * 3
        channels = [4, 16, 32]
        dropouts = [0.1] * 3
        
        ls = [
            nn.Conv1d(in_channels, channels[0], kernel_sizes[0], kernel_strides[0]),
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
            
            ls.append(nn.Flatten()),
            ls.append(nn.Linear(10 * channels[2], embeddim))
        ]
            
        self.backbone = nn.Sequential(*ls)
    
    
    def forward(self, x):
        x = self.backbone(x)
        return x


class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same', dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same', dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvNet(nn.Module):
    '''
    dilated residual network
    '''
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.backbone = nn.Sequential(*[
            DilatedConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.backbone(x)
    
    
class TSEncoder(nn.Module):
    '''
    a time series model with dilated convolutional layers
    '''
    def __init__(self, in_channels=12, hid_channels=64, out_channels=256, depth=10, keep_dim=False):
        super().__init__()
        self.in_dims = in_channels
        self.out_dims = out_channels
        self.hid_dims = hid_channels
        self.proj = nn.Linear(in_channels, hid_channels)
        self.backbone = DilatedConvNet(hid_channels, [hid_channels] * depth + [out_channels], 3)
        self.dropout = nn.Dropout(0.1)
        self.maxpool = nn.AdaptiveMaxPool1d(1) if not keep_dim else None
        
        
    def forward(self, x, mask=False):
        x = x.transpose(1, 2)
        x = self.proj(x)
        
        if mask:
            mask = self._generate_mask(x.shape[0], x.shape[1]).to(x.device)
            x = torch.masked_fill(x, ~mask, 0)
        
        x = x.transpose(1, 2)
        
        x = self.dropout(self.backbone(x))
        
        if self.maxpool is not None:
            x = self.maxpool(x)
            x = x.squeeze(-1)
            
        return x
        
        
    def _generate_mask(self, B, T, n=5, l=0.1):
        mask = torch.ones((B, T))
        if isinstance(n, float):
            n = int(n * T)
        n = max(min(n, T // 2), 1)
        
        if isinstance(l, float):
            l = int(l * T)
        l = max(l, 1)
        
        for i in range(B):
            for _ in range(n):
                t = torch.randint(T-l+1, (1,)).item()
                # For a continuous timestamps, mask all channels
                mask[i, t:t+l] = 0
        return mask.bool()