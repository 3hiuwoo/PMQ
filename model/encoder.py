import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder

def generate_continuous_mask(B, T, C=None, n=5, l=0.1):
    if C:
        res = torch.full((B, T, C), True, dtype=torch.bool)
    else:
        res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            if C:
                # For a continuous timestamps, mask random half channels
                index = np.random.choice(C, int(C/2), replace=False)
                res[i, t:t + l, index] = False
            else:
                # For a continuous timestamps, mask all channels
                res[i, t:t+l] = False
    return res


def generate_binomial_mask(B, T, C=None, p=0.5):
    if C:
        return torch.from_numpy(np.random.binomial(1, p, size=(B, T, C))).to(torch.bool)
    else:
        return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class ProjectionHead(nn.Module):
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
    def __init__(self, input_dims, output_dims, depth, p_output_dims, hidden_dims=64, p_hidden_dims=128,
                 device='cuda', multi_gpu=True):
        super().__init__()
        self.input_dims = input_dims  # Ci
        self.output_dims = output_dims  # Co
        self.hidden_dims = hidden_dims  # Ch
        self.p_hidden_dims = p_hidden_dims  # Cph
        self.p_output_dims = p_output_dims  # Cp
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        # projection head for finetune
        self.proj_head = ProjectionHead(output_dims, p_output_dims, p_hidden_dims)
        device = torch.device(device)
        if device == torch.device('cuda') and multi_gpu:
            self._net = nn.DataParallel(self._net)
            self.proj_head = nn.DataParallel(self.proj_head)
        self._net.to(device)
        self.proj_head.to(device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)


    def forward(self, x):
        out = self.net(x, pool='max')  # B x Co
        x = self.proj_head(out)  # B x Cp
        if self.p_output_dims == 2:  # binary or multi-class
            return torch.sigmoid(x)
        else:
            return x
        

class FTClassifier2(nn.Module):
    def __init__(self, input_dims, output_dims, depth, p_output_dims, hidden_dims=64, p_hidden_dims=128,
                 device='cuda', multi_gpu=True):
        super().__init__()
        self.input_dims = input_dims  # Ci
        self.output_dims = output_dims  # Co
        self.hidden_dims = hidden_dims  # Ch
        self.p_hidden_dims = p_hidden_dims  # Cph
        self.p_output_dims = p_output_dims  # Cp
        self._net = TFEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        # projection head for finetune
        self.proj_head = ProjectionHead(output_dims, p_output_dims, p_hidden_dims)
        device = torch.device(device)
        if device == torch.device('cuda') and multi_gpu:
            self._net = nn.DataParallel(self._net)
            self.proj_head = nn.DataParallel(self.proj_head)
        self._net.to(device)
        self.proj_head.to(device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)


    def forward(self, xt, xf):
        out = self.net(xt, xf, pool=True)  # B x Co
        x = self.proj_head(out)  # B x Cp
        if self.p_output_dims == 2:  # binary or multi-class
            return torch.sigmoid(x)
        else:
            return x


class TFPClassifier(nn.Module):
    def __init__(self, input_dims, output_dims, depth, p_output_dims, hidden_dims=64, p_hidden_dims=128,
                 device='cuda', multi_gpu=True, pool='max'):
        super().__init__()
        self.input_dims = input_dims  # Ci
        self.output_dims = output_dims  # Co
        self.hidden_dims = hidden_dims  # Ch
        self.p_hidden_dims = p_hidden_dims  # Cph
        self.p_output_dims = p_output_dims  # Cp
        self._net_t = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        self._net_f = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        self._proj = MLP(input_dims=output_dims*2, output_dims=output_dims, hidden_dims=(output_dims + output_dims//2))
        self.proj_head = ProjectionHead(output_dims, p_output_dims, p_hidden_dims)
        self.pool = pool
        device = torch.device(device)
        if device == torch.device('cuda') and multi_gpu:
            self._net_t = nn.DataParallel(self.net_t)
            self._net_f = nn.DataParallel(self.net_f)
            self._proj = nn.DataParallel(self.proj)
            self.proj_head = nn.DataParallel(self.proj_head)
        self._net_t.to(device)
        self._net_f.to(device)
        self._proj.to(device)
        self.proj_head.to(device)
        self.net_t = torch.optim.swa_utils.AveragedModel(self._net_t)
        self.net_t.update_parameters(self._net_t)
        self.net_f = torch.optim.swa_utils.AveragedModel(self._net_f)
        self.net_f.update_parameters(self._net_f)
        self.proj = torch.optim.swa_utils.AveragedModel(self._proj)
        self.proj.update_parameters(self._proj)


    def forward(self, xt, xf):
        out_t = self.net_t(xt)
        out_f = self.net_f(xf)
        out = self.proj(torch.cat((out_t, out_f), dim=-1))
        if self.pool == 'max':
            x = F.max_pool1d(out.transpose(1, 2), kernel_size=out.size(1)).squeeze(-1)
        elif self.pool == 'avg':
            x = F.avg_pool1d(out.transpose(1, 2), kernel_size=out.size(1)).squeeze(-1)
        else:
            raise ValueError(f'\'{self.pool}\' is a wrong argument for pool function!')
        x = self.proj_head(x)

        if self.p_output_dims == 2:  # binary or multi-class
            return torch.sigmoid(x)
        else:
            return x
        
        
class TFPClassifier2(nn.Module):
    def __init__(self, input_dims, output_dims, depth, p_output_dims, hidden_dims=64, p_hidden_dims=128,
                 device='cuda', multi_gpu=True, pool='max'):
        super().__init__()
        self.input_dims = input_dims  # Ci
        self.output_dims = output_dims  # Co
        self.hidden_dims = hidden_dims  # Ch
        self.p_hidden_dims = p_hidden_dims  # Cph
        self.p_output_dims = p_output_dims  # Cp
        self._net_t = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        self._net_f = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        self._proj = MLP(input_dims=output_dims*2, output_dims=output_dims, hidden_dims=(output_dims + output_dims//2))
        self.proj_head = ProjectionHead(output_dims, p_output_dims, p_hidden_dims)
        self.pool = pool
        device = torch.device(device)
        if device == torch.device('cuda') and multi_gpu:
            self._net_t = nn.DataParallel(self.net_t)
            self._net_f = nn.DataParallel(self.net_f)
            self._proj = nn.DataParallel(self.proj)
            self.proj_head = nn.DataParallel(self.proj_head)
        self._net_t.to(device)
        self._net_f.to(device)
        self._proj.to(device)
        self.proj_head.to(device)
        self.net_t = torch.optim.swa_utils.AveragedModel(self._net_t)
        self.net_t.update_parameters(self._net_t)
        self.net_f = torch.optim.swa_utils.AveragedModel(self._net_f)
        self.net_f.update_parameters(self._net_f)
        self.proj = torch.optim.swa_utils.AveragedModel(self._proj)
        self.proj.update_parameters(self._proj)


    def forward(self, xt, xf):
        out_t = self.net_t(xt, pool=self.pool)
        out_f = self.net_f(xf, pool=self.pool)
        out = self.proj(torch.cat((out_t, out_f), dim=-1))
        x = self.proj_head(out)

        if self.p_output_dims == 2:  # binary or multi-class
            return torch.sigmoid(x)
        else:
            return x

class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='all_true'):
        super().__init__()
        self.input_dims = input_dims  # Ci
        self.output_dims = output_dims  # Co
        self.hidden_dims = hidden_dims  # Ch
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],  # a list here
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        
        
    def forward(self, x, mask=None, pool=None):  # input dimension : B x O x Ci
        x = self.input_fc(x)  # B x O x Ch (hidden_dims)
        
        # generate & apply mask, default is binomial
        if mask is None:
            # mask should only use in training phase
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'channel_binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1), x.size(2)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'channel_continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1), x.size(2)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        else:
            raise ValueError(f'\'{mask}\' is a wrong argument for mask function!')

        # mask &= nan_masK
        # ~ works as operator.invert
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x O
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x O
        
        if pool == 'max':
            x = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)
        elif pool == 'avg':
            x = F.avg_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)
        else:
            x = x.transpose(1, 2)  # B x O x Co
        
        return x
    
      
class TFEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims  # Ci
        self.output_dims = output_dims  # Co
        self.hidden_dims = hidden_dims  # Ch
        self.mask_mode = mask_mode
        self.input_fc_t = nn.Linear(input_dims, int(hidden_dims/2))
        self.input_fc_f = nn.Linear(input_dims, int(hidden_dims/2))
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],  # a list here
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        
        
    def forward(self, xt, xf, mask=None, pool=True):  # input dimension : B x O x Ci
        xt = self.input_fc_t(xt) # B x O x Ch (hidden_dims)
        xf = self.input_fc_f(xf) # B x O x Ch (hidden_dims)
        # x = xt + xf
        x = torch.cat((xt, xf), dim=-1) # B x O x 2Ch
        
        # generate & apply mask, default is binomial
        if mask is None:
            # mask should only use in training phase
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'channel_binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1), x.size(2)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'channel_continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1), x.size(2)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        else:
            raise ValueError(f'\'{mask}\' is a wrong argument for mask function!')

        # mask &= nan_masK
        # ~ works as operator.invert
        x[~mask] = 0

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x O
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x O
        
        if pool:
            x = F.max_pool1d(x, kernel_size=x.size(-1)).squeeze(-1)
        else:
            x = x.transpose(1, 2)  # B x O x Co
        
        return x
    

        