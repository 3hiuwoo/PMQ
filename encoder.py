'''
This file contains the implementation of the encoder for pretraining and the classifier for finetuning.
'''
import torch
from torch import nn
import torch.nn.functional as F
from utils import generate_binomial_mask, freq_perturb

class SamePadConv(nn.Module):
    ''' Conv1d layer with same padding
    Args:
        in_channels (int): number of input channels, i.e. feature dimension
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        dilation (int): spacing between kernel elements
        groups (int): number of blocked connections from input channels to output channels
    '''
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    ''' Conv1d block with same padding and GELU activation
    Args:
        in_channels (int): number of input channels, i.e. feature dimension
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        dilation (int): spacing between kernel elements
        final (bool): whether this block is the final block
    '''
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    ''' Dilated Conv1d Encoder consisting of multiple ConvBlocks
    Args:
        in_channels (int): number of input channels, i.e. feature dimension
        channels (list): number of output channels for each ConvBlock
        kernel_size (int): size of all the convolving kernel
    '''
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
     
       
    def forward(self, x):
        return self.net(x)
    
    
class ProjectionHead(nn.Module):
    ''' Classfication head for finetuning
    Args:
        input_dims (int): number of input dimensions
        output_dims (int): number of output dimensions
        hidden_dims (int): number of hidden dimensions
    '''
    def __init__(self, input_dims, output_dims, hidden_dims=128):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims

        # projection head for finetune
        self.proj_head = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims)
        )

        self.repr_dropout = nn.Dropout(p=0.1)


    def forward(self, x):
        x = self.repr_dropout(self.proj_head(x))
        if self.output_dims == 2:  # binary or multi-class
            return torch.sigmoid(x)
        else:
            return x


def MLP(input_dims, output_dims, nlayers=1, hidden_dims=320):
    ''' Projection head or Prediction head for pretraining
    Args:
        input_dims (int): number of input dimensions
        output_dims (int): number of output dimensions
        nlayers (int): number of layers
        hidden_dims (int): number of hidden dimensions
    '''
    layers = []
    for i in range(nlayers):
        layers.append(nn.Linear(input_dims, hidden_dims))
        layers.append(nn.BatchNorm1d(hidden_dims))
        layers.append(nn.ReLU())
        input_dims = hidden_dims
    layers.append(nn.Linear(hidden_dims, output_dims))
    layers.append(nn.BatchNorm1d(output_dims))
    return nn.Sequential(*layers)
    

class FTClassifier(nn.Module):
    ''' Classifier for finetuning
    Args:
        input_dims (int): number of input dimensions
        output_dims (int): number of output dimensions of encoder
        hidden_dims (int): number of ouput dimensions of input projector
        depth (int): number of layers for encoder
        p_hidden_dims (int): number of hidden dimensions of projection head
        p_output_dims (int): number of output dimensions of projection head
        pool (str): pooling method for encoder
        device (str): device to run the model
        multi_gpu (bool): whether to use multi-gpu
    '''
    def __init__(self, input_dims, output_dims=320, hidden_dims=64, depth=10, p_hidden_dims=128, p_output_dims=320, pool='avg', device='cuda', multi_gpu=True):
        super().__init__()
        self.input_dims = input_dims  # Ci
        self.output_dims = output_dims  # Co
        self.hidden_dims = hidden_dims  # Ch
        self.p_hidden_dims = p_hidden_dims  # Cph
        self.p_output_dims = p_output_dims  # Cp
        self.pool = pool
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth, pool=pool)
        self.proj_head = ProjectionHead(output_dims, p_output_dims, p_hidden_dims)
        device = torch.device(device)
        if device == torch.device('cuda') and multi_gpu:
            self._net = nn.DataParallel(self._net)
            self.proj_head = nn.DataParallel(self.proj_head)
        self._net.to(device)
        self.proj_head.to(device)
        # just for matching the key names
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)


    def forward(self, x):
        out = self.net(x)  # B x Co
        x = self.proj_head(out)  # B x Cp
        if self.p_output_dims == 2:  # binary or multi-class
            return torch.sigmoid(x)
        else:
            return x
    
    
class TSEncoder(nn.Module):
    ''' Encoder for pretraining
    Args:
        input_dims (int): number of input dimensions
        output_dims (int): number of output dimensions
        hidden_dims (int): number of output dimensions of input projector
        depth (int): number of layers
        mask_t (float): probability of time mask
        mask_f (float): ratio of freq mask
        pool (str): pooling method for encoder
    '''
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_t=0, mask_f=0, pool='avg'):
        super().__init__()
        self.input_dims = input_dims  # Ci
        self.output_dims = output_dims  # Co
        self.hidden_dims = hidden_dims  # Ch
        self.mask_t = mask_t
        self.mask_f = mask_f
        self.pool = pool
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],  # a list here
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        
        
    def forward(self, x):  # input dimension : B x O x Ci
        x = self.input_fc(x)  # B x O x Ch (hidden_dims)
        
        if self.mask_f > 0:
            x = freq_perturb(x, self.mask_f)
        
        if self.mask_t > 0:
            mask = generate_binomial_mask(x.size(0), x.size(1), p=self.mask_t).to(x.device)
            x[~mask] = 0
        # if mask == 'binomial':
        #     mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        # elif mask == 'channel_binomial':
        #     mask = generate_binomial_mask(x.size(0), x.size(1), x.size(2)).to(x.device)
        # elif mask == 'continuous':
        #     mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        # elif mask == 'channel_continuous':
        #     mask = generate_continuous_mask(x.size(0), x.size(1), x.size(2)).to(x.device)
        # elif mask == 'all_true':
        #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        # elif mask == 'all_false':
        #     mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        # elif mask == 'mask_last':
        #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        #     mask[:, -1] = False
        # else:
        #     raise ValueError(f'\'{mask}\' is a wrong argument for mask function!')
        
        x = x.transpose(1, 2)  # B x Ch x O
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x O
        
        if self.pool == 'max':
            x = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1) # B x Co
        elif self.pool == 'avg':
            x = F.avg_pool1d(x, kernel_size=x.size(-1)).squeeze(-1) # B x Co
        else:
            x = x.transpose(1, 2)  # B x O x Co
        
        return x