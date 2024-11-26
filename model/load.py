import torch
import re
import numpy as np
from torch import nn
from model.base import CNN3


def load_network(network_name):
    '''
    return backbone network class
    '''
    if network_name == 'cnn3':
        return CNN3
    else:
        raise ValueError(f'Unknown network {network_name}')


def load_model(model_name, task, embeddim):
    '''
    load model
    '''
    network = load_network(model_name)
    
    if task in ['cmsc', 'simclr']:
        return ContrastModel(network=network, embeddim=embeddim)
    elif task == 'moco':
        return MoCoModel(network=network, embeddim=embeddim)
    elif task == 'mcp':
        return MCPModel(network=network, embeddim=embeddim)
    elif task == 'supervised':
        return SupervisedModel(network=network, embeddim=embeddim)
    else:
        raise ValueError(f'Unknown task {task}')
        

class ContrastModel(nn.Module):
    '''
    contrastive model used for CMSC, SimCLR
    '''
    def __init__(self, network, embeddim=256):
        super(ContrastModel, self).__init__()
        self.embeddim = embeddim
        self.encoder = network(embeddim)
        
        
    def forward(self, x):
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
    '''
    supervised model
    '''
    def __init__(self, network, num_classes=4, embeddim=256):
        super(SupervisedModel, self).__init__()
        self.embeddim = embeddim
        self.encoder = network(embeddim)
        # dim = self.encoder.fc.weight.shape[1]
        # self.encoder.fc = nn.Linear(dim, num_classes)
        self.fc = nn.Linear(embeddim, num_classes)
        
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x
    
    
class MoCoModel(nn.Module):
    '''
    MoCo model
    '''
    def __init__(self, network, embeddim=256, queue_size=16384, momentum=0.999):
        super(MoCoModel, self).__init__()
        self.embeddim = embeddim
        self.encoder_q = network(embeddim)
        self.encoder_k = network(embeddim)
        self.queue_size = queue_size
        self.momentum = momentum
        self.register_buffer("queue", torch.randn(embeddim, queue_size))
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
    def __init__(self, network, embeddim=256, queue_size=16384, momentum=0.999):
        super(MCPModel, self).__init__()
        self.embeddim = embeddim
        self.encoder_q = network(embeddim)
        self.encoder_k = network(embeddim)
        self.queue_size = queue_size
        self.momentum = momentum
        self.register_buffer("queue", torch.randn(embeddim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("heads", torch.zeros(queue_size, dtype=torch.long))
        
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
    def _update_queue_n_head(self, keys, heads):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:(ptr+batch_size)] = keys.T
        self.heads[ptr:(ptr+batch_size)] = heads
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr
        
        
    @torch.no_grad()
    def _strip_heads(self, heads):
        strip_heads = []
        for head in heads:
            pattern = re.search(r'\d+$', head)
            strip_heads.append(int(pattern.group()))
        return strip_heads
            
            
    def forward(self, x, heads):
        """
        Input:
            x: input with 2 views (Bx2xCxS)
        Output:
            logits
        """
        x = x.permute(1, 0, 2, 3)
        current_heads = self._strip_heads(heads)
        current_heads = torch.tensor(current_heads, dtype=torch.long, device=self.heads.device)
        
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
        
        queue_heads = self.heads.clone().detach()
        
        # dequeue and enqueue
        self._update_queue_n_head(k, current_heads)

        return query_key, query_queue, queue_heads, current_heads