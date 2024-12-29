import torch
import re
import numpy as np
from torch import nn
from model.base import TSEncoder


def load_model(model_name, task, in_channels=12, out_channels=256, depth=10, num_classes=4):
    '''
    load model
    '''
    if task in ['cmsc', 'simclr']:
        return ContrastModel(network=TSEncoder, in_channels=in_channels, out_channels=out_channels, depth=depth)
    elif task == 'comet':
        return COMETModel(network=TSEncoder, in_channels=in_channels, out_channels=out_channels, depth=depth)
    elif task == 'moco':
        return MoCoModel(network=TSEncoder, in_channels=in_channels, out_channels=out_channels, depth=depth)
    elif task == 'mcp':
        return MCPModel(network=TSEncoder, in_channels=in_channels, out_channels=out_channels, depth=depth)
    elif task == 'supervised':
        return SupervisedModel(network=TSEncoder, in_channels=in_channels, num_classes=num_classes,
                               out_channels=out_channels, depth=depth)
    else:
        raise ValueError(f'Unknown task {task}')
    
    
class SupervisedModel(nn.Module):
    '''
    supervised model
    '''
    def __init__(self, network, in_channels=1, num_classes=4, out_channels=256, depth=10):
        super(SupervisedModel, self).__init__()
        self.out_channels = out_channels
        self.encoder = network(in_channels=in_channels, out_channels=out_channels, depth=depth)
        self.fc = nn.Linear(out_channels, num_classes)
        
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x        

class ContrastModel(nn.Module):
    '''
    contrastive model used for CMSC, SimCLR
    '''
    def __init__(self, network, in_channels=1, out_channels=256, depth=10):
        super(ContrastModel, self).__init__()
        self.out_channels = out_channels
        self.encoder = network(in_channels=in_channels, out_channels=out_channels, depth=depth)
        
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): inputs with N views (BxNxCxT)
        Returns:
            h (torch.Tensor): latent embedding for each of the N views (NxBxH)
        """
        nviews = x.shape[1]
        x = x.permute(1, 0, 2, 3)
        h = [self.encoder(x[n, ...]) for n in range(nviews)]
        return torch.stack(h, dim=0)


class COMETModel(nn.Module):
    '''
    COMET model
    '''
    def __init__(self, network, in_channels=12, out_channels=256, depth=10):
        super(COMETModel, self).__init__()
        self.out_channels = out_channels
        self.encoder = network(in_channels=in_channels, out_channels=out_channels, depth=depth, keep_dim=True)
        
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): raw inputs with shape BxCxT
        Returns:
            h (torch.Tensor): latent embedding for (observation, sample, trial, patient) levels, each with 2 views.
        """
        nlevels = 4
        nviews = 2
        mask = [True, True, False, False]
        
        ls = []
        for l in range(nlevels):
            h = torch.stack([self.encoder(x, mask=mask[l]) for _ in range(nviews)], dim=0)
            ls.append(h)
        return torch.stack(ls, dim=0)
  
  
class MoCoModel(nn.Module):
    '''
    MoCo model
    '''
    def __init__(self, network, in_channels=1, out_channels=256, depth=10, queue_size=65536, momentum=0.999):
        super(MoCoModel, self).__init__()
        self.out_channels = out_channels
        self.encoder_q = network(in_channels=in_channels, out_channels=out_channels, depth=depth)
        self.encoder_k = network(in_channels=in_channels, out_channels=out_channels, depth=depth)
        self.queue_size = queue_size
        self.momentum = momentum
        self.register_buffer("queue", torch.randn(out_channels, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False


    @torch.no_grad()
    def _update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)


    @torch.no_grad()
    def _update_queue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr
        

    def forward(self, x):
        """
        Input:
            x: input with 2 views (Bx2xCxS)
        Output:
            logits
        """
        x = x.permute(1, 0, 2, 3)
        
        # compute query features
        q = self.encoder_q(x[0])  # queries: BxH
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            idx = torch.randperm(x[1].size(0), device=x.device)
            k = self.encoder_k(x[1, idx, ...])  # keys: BxH
            
            # undo shuffle
            k = k[torch.argsort(idx)]
            k = nn.functional.normalize(k, dim=1)

        # positive logits: Nx1
        pos = torch.einsum("nq,nq->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        neg = torch.einsum("nq,qk->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([pos, neg], dim=1)

        # dequeue and enqueue
        self._update_queue(k)

        return logits


class MCPModel(nn.Module):
    '''
    MoCo model patient specific variant
    '''
    def __init__(self, network, in_channels=1, out_channels=256, depth=10, queue_size=16384, momentum=0.999):
        super(MCPModel, self).__init__()
        self.out_channels = out_channels
        self.encoder_q = network(in_channels=in_channels, out_channels=out_channels, depth=depth)
        self.encoder_k = network(in_channels=in_channels, out_channels=out_channels, depth=depth)
        self.queue_size = queue_size
        self.momentum = momentum
        self.register_buffer("queue", torch.randn(out_channels, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False


    @torch.no_grad()
    def _update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)


    @torch.no_grad()
    def _update_queue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr
        
                    
    def forward(self, x):
        """
        Input:
            x: input with 2 views (Bx2xCxS)
            queue_heads: patient id queue passed in from the training loop
        Output:
            logits
        """
        x = x.permute(1, 0, 2, 3)
        
        # compute query features
        q = self.encoder_q(x[0])  # queries: BxH
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            idx = torch.randperm(x[1].size(0), device=x.device)
            k = self.encoder_k(x[1, idx, ...])  # keys: BxH
            
            # undo shuffle
            k = k[torch.argsort(idx)]
            k = nn.functional.normalize(k, dim=1)

        query_key = torch.matmul(q, k.T) # BxB
        query_queue = torch.matmul(q, self.queue.clone().detach()) # BxK

        # dequeue and enqueue
        self._update_queue(k)

        return query_key, query_queue