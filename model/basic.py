from torch import nn

class CNN3(nn.Module):
    '''
    a convolutional neural network with 3 convolutional layers
    '''
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
    
    



