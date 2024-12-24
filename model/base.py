from torch import nn
from torch.nn import functional as F

class CNN3(nn.Module):
    '''
    a convolutional neural network with 3 convolutional layers
    '''
    def __init__(self, embeddim=256):
        super(CNN3, self).__init__()
        
        kernel_sizes = [7] * 3
        kernel_strides = [3] * 3
        channels = [4, 16, 32]
        dropouts = [0.1] * 3
        
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


class Residual(nn.Module):
    '''
    Residual block with feature dimension kept the same
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding='same', final=False):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, dilation)
        
        if in_channels != out_channels or final:
            self.conv3 = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.conv3 = None
            
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)


    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        
        if self.conv3:
            X = self.conv3(X)  
        Y += X
        
        return F.relu(Y)
    
    
class Res20(nn.Module):
    '''ResNet with 10 residual block, each with 2 convolutional layers
    Args:
        channels(list): list of number of channels, the first element is the number of input channels,
                        others are output channels for each residual block
        kernel_size(int): the size of the every kernel
        avepool(bool): whether to add global average pooling layer at the end     
    '''
    def __init__(self, in_channels, embeddim=256, keep_dim=False):
        super().__init__()
        
        channels = [in_channels] + [64] * 9 + [embeddim]
        
        blk_list = [
            Residual(
                channels[i],
                channels[i+1],
                kernel_size=3,
                # dilation=2**i,
                final=(i == len(channels)-2)
            )
            for i in range(len(channels)-1)
        ]
        blk_list.append(nn.Dropout1d(0.1))
        
        if not keep_dim:
            blk_list.append(nn.AdaptiveAvgPool1d(1))
            blk_list.append(nn.Flatten())
            
        self.backbone = nn.Sequential(*blk_list)
        
        
    def forward(self, x):
        return self.backbone(x)


