import torch
from torch import nn
from model.basic import CNN3


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
    elif task == 'supervised':
        return SupervisedModel(network=network, embeddim=embeddim)
    else:
        raise ValueError(f'Unknown task {task}')
        

class ContrastModel(nn.Module):
    '''
    contrastive model used for CMSC, SimCLR
    '''
    def __init__(self, network, embeddim=256):
        super(ContrastModel,self).__init__()
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
        super(SupervisedModel,self).__init__()
        self.embeddim = embeddim
        self.encoder = network(embeddim)
        dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Linear(dim, num_classes)
        
    
    def forward(self, x):
        x = self.encoder(x)
        return x
    
    
class MoCoModel(nn.Module):
    '''
    MoCo model
    '''
    def __init__(self, network, embeddim=256, queue_size=16384, momentum=0.999):
        super(MoCoModel,self).__init__()
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
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle


    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


    def forward(self, x):
        """
        Input:
            x: input with 2 views (Bx2xCxS)
        Output:
            logits, targets
        """
        x = x.permute(1, 0, 2, 3)
        
        # compute query features
        q = self.encoder_q(x[0])  # queries: BxH
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: BxH
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= 0.1

        # dequeue and enqueue
        self._update_queue(k)

        return logits


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output