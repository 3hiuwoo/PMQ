import os
from datetime import datetime
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from model.encoder import TSEncoder, ProjectionHead
from model.loss import id_momentum_contrastive_loss as loss_func
from utils import shuffle_feature_label, MyBatchSampler

class TFP:
    ''' A momentum contrastive learning model cross time, frequency, patient.
    
    Args:
        input_dims (int): The input dimension. For a uni-variate time series, this should be set to 1.
        output_dims (int): The representation dimension.
        hidden_dims (int): The hidden dimension of the encoder.
        proj_dims (int): The hidden and output dimension of the projection head, pass None to disable appending projection head to encoder.
        depth (int): The number of hidden residual blocks in the encoder.
        device (str): The gpu used for training and inference.
        lr (float): The learning rate.
        batch_size (int): The batch size of samples.
        momentum (float): The momentum used for the key encoder.
        queue_size (int): The size of the queue.
        multi_gpu (bool): A flag to indicate whether using multiple gpus
    '''
    def __init__(
        self,
        input_dims=12,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        pool=None,
        device='cuda',
        lr=1e-4,
        batch_size=256,
        momentum=0.999,
        queue_size=65536,
        multi_gpu=False,
    ):
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        
        self.multi_gpu = multi_gpu
        
        self.momentum = momentum
        self.queue_size = queue_size
        
        self._net_t = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        self._net_f = TSEncoder(input_dims=input_dims//2+1, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        self._proj = ProjectionHead(input_dims=output_dims*2, output_dims=output_dims, hidden_dims=(output_dims + output_dims//2), dropout=0)
        
        self.momentum_net_t = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        self.momentum_net_f = TSEncoder(input_dims=input_dims//2+1, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        self.momentum_proj = ProjectionHead(input_dims=output_dims*2, output_dims=output_dims, hidden_dims=(output_dims + output_dims//2), dropout=0)
        
        self._momentum_init() # initialize all momentum parts
                
        device = torch.device(device)
        if device == torch.device('cuda') and self.multi_gpu:
            self._net_t = nn.DataParallel(self._net_t)
            self._net_f = nn.DataParallel(self._net_f)
            self.momentum_net_t = nn.DataParallel(self.momentum_net_t)
            self.momentum_net_f = nn.DataParallel(self.momentum_net_f)
            self.proj = nn.DataParallel(self.proj)
            self.momentum_proj = nn.DataParallel(self.momentum_proj)
                
        self._net_t.to(device)
        self._net_f.to(device)
        self._proj.to(device)
        self.momentum_net_t.to(device)
        self.momentum_net_f.to(device)
        self.momentum_proj.to(device)
        
        self.net_t = torch.optim.swa_utils.AveragedModel(self._net_t)
        self.net_t.update_parameters(self._net_t)
        self.net_f = torch.optim.swa_utils.AveragedModel(self._net_f)
        self.net_f.update_parameters(self._net_f)
        self.proj = torch.optim.swa_utils.AveragedModel(self._proj)
        self.proj.update_parameters(self._proj)
        
        self.pool = pool
        if pool is None:
            self.queue = torch.randn(queue_size, 300*output_dims, device=device, requires_grad=False)
            self.queue = F.normalize(self.queue, dim=1)
        else:
            self.queue = torch.randn(queue_size, output_dims, device=device, requires_grad=False)
            self.queue = F.normalize(self.queue, dim=1)
        
        self.id_queue = torch.zeros(queue_size, dtype=torch.long, device=device, requires_grad=False)
        self.queue_ptr = torch.zeros(1, dtype=torch.long, device=device, requires_grad=False)
        
    
    def _momemtum_init(self):
        for param_q, param_k in zip(self._net_t.parameters(), self.momentum_net_t.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
        for param_q, param_k in zip(self._net_f.parameters(), self.momentum_net_f.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
        for param_q, param_k in zip(self.proj.parameters(), self.momentum_proj.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
            
    def fit(self, X, y, shuffle_function='random', mask='all_true', epochs=None, schedule=None, logdir='', checkpoint=1, verbose=1):
        ''' Training the TFP model.
        
        Args:
            X (numpy.ndarray): The training data. It should have a shape of (n_samples, sample_timestamps, features).
            y (numpy.ndarray): The training labels. It should have a shape of (n_samples, 3). The three columns are the label, patient id, and trial id.
            shuffle_function (str): specify the shuffle function.
            mask_type (str): A list of masking functions applied (str).
            epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            verbose (int): Whether to print the training loss after each epoch.
            
        Returns:
            epoch_loss_list: a list containing the training losses on each epoch.
        '''
        assert X.ndim == 3 # X.shape = (total_size, length, channels)
        assert y.shape[1] == 3
        
        if X.shape.index(min(X.shape)) == 1:
            print('=> Transpose X to have the last dimension of feature size')
            X = X.transpose(0, 2, 1)

        if shuffle_function == 'trial':
            X, y = shuffle_feature_label(X, y, shuffle_function=shuffle_function, batch_size=self.batch_size)

        # we need patient id for patient-level contrasting and trial id for trial-level contrasting
        train_dataset = TensorDataset(
            torch.from_numpy(X).to(torch.float),
            torch.from_numpy(y).to(torch.long)
            )
        
        if shuffle_function == 'random':
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        else:
            print('=> Shuffle data by trial')
            my_sampler = MyBatchSampler(range(len(train_dataset)), batch_size=self.batch_size, drop_last=True)
            train_loader = DataLoader(train_dataset, batch_sampler=my_sampler)
        
        params = list(self._net_t.parameters()) + list(self._net_f.parameters()) + list(self._proj.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr)
        scheduler = self.get_scheduler(schedule, optimizer, epochs)
        if scheduler:
            print(f'=> Using scheduler: {schedule}')
        
        if mask == 'all_true':
            diag = False
            print('=> Diagonal elements are not considered in loss calculation')
            
        epoch_loss_list = []
            
        start_time = datetime.now()  
        for epoch in range(epochs):
            cum_loss = 0
            for x, y in tqdm(train_loader, desc=f'=> Epoch {epoch+1}', leave=False):
                x = x.to(self.device)
                pid = y[:, 1]  # patient id

                self._momentum_update()
                    
                optimizer.zero_grad()
                
                s = torch.fft.rfft(x, dim=1, n=2*x.size(1)-1, norm='ortho').abs()
                
                ht = self._net_t(x, mask='all_true', pool=self.pool)
                hf = self._net_f(s, mask='all_true', pool=self.pool)
                h = torch.cat([ht, hf], dim=-1)
                z = self.proj(h)
                
                htm = self.momentum_net_t(x, mask=mask, pool=self.pool)
                hfm = self.momentum_net_f(x, mask=mask, pool=self.pool)
                hm = torch.cat([htm, hfm], dim=-1)
                zm = self.momentum_proj(hm)
                
                if self.pool is None:
                    q = nn.functional.normalize(z, dim=-1)
                    k = nn.functional.normalize(zm, dim=-1)
                    q = q.permute(0, 2, 1).reshape((q.size(0), -1))
                    k = k.permute(0, 2, 1).reshape((k.size(0), -1))
                    q = nn.functional.normalize(q, dim=1)
                    k = nn.functional.normalize(k, dim=1)
                else:
                    q = nn.functional.normalize(z, dim=-1)
                    k = nn.functional.normalize(zm, dim=-1)
                    
                loss = loss_func(z, zm, self.queue.clone().detach, pid, self.id_queue.clone().detach(), diag=diag)
                
                loss.backward()
                optimizer.step()
                self._update_swa()

                cum_loss += loss.item()
                
                self._update_queue(k, pid)

            cum_loss /= len(train_loader)
            epoch_loss_list.append(cum_loss)
            
            if schedule == 'plateau':
                scheduler.step(cum_loss)
            elif scheduler:
                scheduler.step()
            
            if verbose:
                print(f"=> Epoch {epoch+1}: loss: {cum_loss}")
                
            if (epoch+1) % checkpoint == 0:
                self.save(os.path.join(logdir, f'pretrain_{epoch+1}.pth'))
                
        end_time = datetime.now()
        print(f'=> Training finished in {end_time - start_time}')
            
        return epoch_loss_list
        
    
    def _update_swa(self):
        self.net_t.update_parameters(self._net_t)
        self.net_f.update_parameters(self._net_f)
        self.proj.update_parameters(self._proj)
        
    
    def _momentum_update(self):
        '''
        Momentum update of the key encoder
        '''
        with torch.no_grad():
            for param_q, param_k in zip(
                self.net_t.parameters(), self.momentum_net_t.parameters()
            ):
                param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
            
            for param_q, param_k in zip(
                self.net_f.parameters(), self.momentum_net_f.parameters()
            ):
                param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
                
            for param_q, param_k in zip(
                self.proj.parameters(), self.momentum_proj.parameters()
            ):
                param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
                
    
    def _update_queue(self, k, pid):
        '''
        update all queues
        '''
        # gather keys before updating queue
        batch_size = k.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr : ptr + batch_size, ...] = k
        self.id_queue[ptr : ptr + batch_size] = pid
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr
        
        
    def get_scheduler(self, schedule, optimizer, epochs):
        if schedule == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
        elif schedule == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        elif schedule == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif schedule == 'cosine_warm':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs//10, T_mult=2)
        else:
            scheduler = None
            
        return scheduler
        
   
    def save(self, fn):
        '''Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        checkpoint = {
            'net_t': self.net_t.state_dict(),
            'net_f': self.net_f.state_dict(),
            'proj': self.proj.state_dict()
        }
        torch.save(checkpoint, fn)
    
    
    def load(self, fn):
        pass
    
