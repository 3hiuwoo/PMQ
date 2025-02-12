import torch
import os
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from model.encoder import TSEncoder, TFEncoder, ProjectionHead
from model.loss import id_momentum_loss, id_momentum_loss2, id_contrastive_loss
from utils import shuffle_feature_label, MyBatchSampler, transform

class MOPA:
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
        proj_dims=256,
        depth=10,
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
        self.proj_dims = proj_dims
        
        self.multi_gpu = multi_gpu
        
        self.momentum = momentum
        self.queue_size = queue_size
        
        self.net_q = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        self.net_k = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        
        self.proj_q = ProjectionHead(input_dims=output_dims, output_dims=output_dims, hidden_dims=proj_dims) if proj_dims else None
        self.proj_k = ProjectionHead(input_dims=output_dims, output_dims=output_dims, hidden_dims=proj_dims) if proj_dims else None

        for param_q, param_k in zip(
            self.net_q.parameters(), self.net_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        if self.proj_k:
            for param_q, param_k in zip(
                self.proj_q.parameters(), self.proj_k.parameters()
            ):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            
        device = torch.device(device)
        if device == torch.device('cuda') and self.multi_gpu:
            # self.net_q = nn.DataParallel(self.net_q, device_ids=gpu_idx_list)
            self.net_q = nn.DataParallel(self.net_q)
            self.net_k = nn.DataParallel(self.net_k)
            if self.proj_q:
                self.proj_q = nn.DataParallel(self.proj_q)
                self.proj_k = nn.DataParallel(self.proj_k)
                
        self.net_q.to(device)
        self.net_k.to(device)
        if self.proj_q:
            self.proj_q.to(device)
            self.proj_k.to(device)
            
        # stochastic weight averaging
        # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        # self.net = self.net_q
        self.net = torch.optim.swa_utils.AveragedModel(self.net_q)
        self.net.update_parameters(self.net_q)

        self.queue = torch.randn(queue_size, output_dims, device=device, requires_grad=False)
        self.queue = F.normalize(self.queue, dim=1)
        
        self.id_queue = torch.zeros(queue_size, dtype=torch.long, device=device, requires_grad=False)
        self.queue_ptr = torch.zeros(1, dtype=torch.long, device=device, requires_grad=False)
        
    
    def fit(self, X, y, shuffle_function='random', mask_type='t+fb', epochs=None, schedule=[30, 80], logdir='', checkpoint=1, verbose=1):
            ''' Training the MoPa model.
            
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
            assert self.queue_size % self.batch_size == 0
            
            if X.shape.index(min(X.shape)) == 1:
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
                my_sampler = MyBatchSampler(range(len(train_dataset)), batch_size=self.batch_size, drop_last=True)
                train_loader = DataLoader(train_dataset, batch_sampler=my_sampler)
            
            if self.proj_q:
                params = list(self.net_q.parameters()) + list(self.proj_q.parameters())
                print(f'=> Append projection head to encoder with dimension: {self.proj_dims}')
            else:
                params = self.net_q.parameters()
                
            optimizer = torch.optim.AdamW(params, lr=self.lr)
            scheduler = self.get_scheduler(schedule, optimizer, epochs)
            if scheduler:
                print(f'=> Using scheduler: {schedule}')
                
            epoch_loss_list = []
            masks = mask_type.split('+') # e.g. 't+fb' -> ['t', 'fb']
            if masks[0] == masks[1] and len(masks[0]) == 1: # e.g. 't+t'
                loss_func = id_momentum_loss2
                print('=> Diagonal loss does not count')
            else:
                loss_func = id_momentum_loss
                
            start_time = datetime.now()  
            for epoch in range(epochs):
                cum_loss = 0
                for x, y in tqdm(train_loader, desc=f'=> Epoch {epoch+1}', leave=False):
                    x = x.to(self.device)
                    # get encoder mask mode and transformed data
                    x1, mask1 = transform(x, opt=masks[0])
                    x2, mask2 = transform(x, opt=masks[1])
                    pid = y[:, 1]  # patient id
                    
                    with torch.no_grad():
                        self._momentum_update_key_encoder()
                        
                    optimizer.zero_grad()
                    
                    q = self.net_q(x1, mask=mask1, pool=True)
                    if self.proj_q:
                        q = self.proj_q(q)
                    q = F.normalize(q, dim=1)
                    
                    with torch.no_grad():
                        # shuffle BN
                        idx = torch.randperm(x2.size(0), device=x.device)
                        k = self.net_k(x2[idx], mask=mask2, pool=True)
                        if self.proj_k:
                            k = self.proj_k(k)
                        k = F.normalize(k, dim=1)
                        k = k[torch.argsort(idx)]
                    
                    loss = loss_func(q, k, self.queue.clone().detach(), pid, self.id_queue.clone().detach())

                    loss.backward()
                    optimizer.step()
                    self.net.update_parameters(self.net_q)

                    cum_loss += loss.item()
                    
                    self._dequeue_and_enqueue(k, pid)

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
        
        
    def _momentum_update_key_encoder(self):
        '''
        Momentum update of the key encoder
        '''
        for param_q, param_k in zip(
            self.net.parameters(), self.net_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
            
        if self.proj_q:
            for param_q, param_k in zip(
                self.proj_q.parameters(), self.proj_k.parameters()
            ):
                param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)


    def _dequeue_and_enqueue(self, keys, pid):
        '''
        update all queues
        '''
        # gather keys before updating queue
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr : ptr + batch_size, ...] = keys
        self.id_queue[ptr : ptr + batch_size] = pid
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr
        
        
    def get_scheduler(self, schedule, optimizer, epochs):
        if schedule == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 80], gamma=0.1)
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
        torch.save(self.net.state_dict(), fn)
    
    
    def load(self, fn):
        '''Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        # state_dict = torch.load(fn, map_location=self.device)
        state_dict = torch.load(fn)
        self.net.load_state_dict(state_dict)
        
        
class MOPA2:
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
        proj_dims=256,
        depth=10,
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
        self.proj_dims = proj_dims
        
        self.multi_gpu = multi_gpu
        
        self.momentum = momentum
        self.queue_size = queue_size
        
        self.net_q = TFEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        self.net_k = TFEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        
        self.proj_q = ProjectionHead(input_dims=output_dims, output_dims=output_dims, hidden_dims=proj_dims) if proj_dims else None
        self.proj_k = ProjectionHead(input_dims=output_dims, output_dims=output_dims, hidden_dims=proj_dims) if proj_dims else None

        for param_q, param_k in zip(
            self.net_q.parameters(), self.net_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        if self.proj_k:
            for param_q, param_k in zip(
                self.proj_q.parameters(), self.proj_k.parameters()
            ):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
            
        device = torch.device(device)
        if device == torch.device('cuda') and self.multi_gpu:
            # self.net_q = nn.DataParallel(self.net_q, device_ids=gpu_idx_list)
            self.net_q = nn.DataParallel(self.net_q)
            self.net_k = nn.DataParallel(self.net_k)
            if self.proj_q:
                self.proj_q = nn.DataParallel(self.proj_q)
                self.proj_k = nn.DataParallel(self.proj_k)
                
        self.net_q.to(device)
        self.net_k.to(device)
        if self.proj_q:
            self.proj_q.to(device)
            self.proj_k.to(device)
            
        # stochastic weight averaging
        # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        # self.net = self.net_q
        self.net = torch.optim.swa_utils.AveragedModel(self.net_q)
        self.net.update_parameters(self.net_q)

        self.queue = torch.randn(queue_size, output_dims, device=device, requires_grad=False)
        self.queue = F.normalize(self.queue, dim=1)
        
        self.id_queue = torch.zeros(queue_size, dtype=torch.long, device=device, requires_grad=False)
        self.queue_ptr = torch.zeros(1, dtype=torch.long, device=device, requires_grad=False)
        
    
    def fit(self, X, y, shuffle_function='random', mask_type='t+s', epochs=None, schedule=[30, 80], logdir='', checkpoint=1, verbose=1):
            ''' Training the MoPa model.
            
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
            assert self.queue_size % self.batch_size == 0
            
            if X.shape.index(min(X.shape)) == 1:
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
            
            if self.proj_q:
                params = list(self.net_q.parameters()) + list(self.proj_q.parameters())
                print(f'=> Append projection head to encoder with dimension: {self.proj_dims}')
            else:
                params = self.net_q.parameters()
                
            optimizer = torch.optim.AdamW(params, lr=self.lr)
            scheduler = self.get_scheduler(schedule, optimizer, epochs)
            if scheduler:
                print(f'=> Using scheduler: {schedule}')
                
            epoch_loss_list = []
            masks = mask_type.split('+') # e.g. 't+fb' -> ['t', 'fb']
            if len(masks[0]) == 1 and len(masks[1]) == 1: # e.g. 't+s'
                loss_func = id_momentum_loss2
                print('=> Diagonal loss does not count')
            else:
                loss_func = id_momentum_loss
                
            start_time = datetime.now()  
            for epoch in range(epochs):
                cum_loss = 0
                for x, y in tqdm(train_loader, desc=f'=> Epoch {epoch+1}', leave=False):
                    x = x.to(self.device)
                    # get encoder mask mode and transformed data
                    x1, mask1 = transform(x, opt=masks[0])
                    x2, mask2 = transform(x, opt=masks[1])
                    pid = y[:, 1]  # patient id
                    
                    with torch.no_grad():
                        self._momentum_update_key_encoder()
                        
                    optimizer.zero_grad()
                    
                    q = self.net_q(x1, x2, mask=mask1, pool=True)
                    if self.proj_q:
                        q = self.proj_q(q)
                    q = F.normalize(q, dim=1)
                    
                    with torch.no_grad():
                        # shuffle BN
                        idx = torch.randperm(x2.size(0), device=x.device)
                        k = self.net_k(x1[idx], x2[idx], mask=mask2, pool=True)
                        if self.proj_k:
                            k = self.proj_k(k)
                        k = F.normalize(k, dim=1)
                        k = k[torch.argsort(idx)]

                    loss = loss_func(q, k, self.queue.clone().detach(), pid, self.id_queue.clone().detach())

                    loss.backward()
                    optimizer.step()
                    self.net.update_parameters(self.net_q)

                    cum_loss += loss.item()
                    
                    self._dequeue_and_enqueue(k, pid)

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
        
        
    def _momentum_update_key_encoder(self):
        '''
        Momentum update of the key encoder
        '''
        for param_q, param_k in zip(
            self.net.parameters(), self.net_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
            
        if self.proj_q:
            for param_q, param_k in zip(
                self.proj_q.parameters(), self.proj_k.parameters()
            ):
                param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)


    def _dequeue_and_enqueue(self, keys, pid):
        '''
        update all queues
        '''
        # gather keys before updating queue
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr : ptr + batch_size, ...] = keys
        self.id_queue[ptr : ptr + batch_size] = pid
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr
        
        
    def get_scheduler(self, schedule, optimizer, epochs):
        if schedule == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 80], gamma=0.1)
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
        torch.save(self.net.state_dict(), fn)
    
    
    def load(self, fn):
        '''Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        # state_dict = torch.load(fn, map_location=self.device)
        state_dict = torch.load(fn)
        self.net.load_state_dict(state_dict)
        

class CMC:
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
        proj_dims=256,
        depth=10,
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
        self.proj_dims = proj_dims
        
        self.multi_gpu = multi_gpu
        
        self.momentum = momentum
        self.queue_size = queue_size
        
        self.net_q = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        self.net_k = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        
        self.proj_q = ProjectionHead(input_dims=output_dims, output_dims=output_dims, hidden_dims=proj_dims) if proj_dims else None
        self.proj_k = ProjectionHead(input_dims=output_dims, output_dims=output_dims, hidden_dims=proj_dims) if proj_dims else None
            
        device = torch.device(device)
        if device == torch.device('cuda') and self.multi_gpu:
            # self.net_q = nn.DataParallel(self.net_q, device_ids=gpu_idx_list)
            self.net_q = nn.DataParallel(self.net_q)
            self.net_k = nn.DataParallel(self.net_k)
            if self.proj_q:
                self.proj_q = nn.DataParallel(self.proj_q)
                self.proj_k = nn.DataParallel(self.proj_k)
                
        self.net_q.to(device)
        self.net_k.to(device)
        if self.proj_q:
            self.proj_q.to(device)
            self.proj_k.to(device)
            
        # stochastic weight averaging
        # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        # self.net = self.net_q
        self.net_t = torch.optim.swa_utils.AveragedModel(self.net_q)
        self.net_t.update_parameters(self.net_q)
        self.net_f = torch.optim.swa_utils.AveragedModel(self.net_k)
        self.net_f.update_parameters(self.net_k)

        # self.queue = torch.randn(queue_size, output_dims, device=device, requires_grad=False)
        # self.queue = F.normalize(self.queue, dim=1)
        
        # self.id_queue = torch.zeros(queue_size, dtype=torch.long, device=device, requires_grad=False)
        # self.queue_ptr = torch.zeros(1, dtype=torch.long, device=device, requires_grad=False)
        
    
    def fit(self, X, y, shuffle_function='trial', mask_type='t+s', epochs=None, schedule=[30, 80], logdir='', checkpoint=1, verbose=1):
            ''' Training the MoPa model.
            
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
            assert self.queue_size % self.batch_size == 0
            
            if X.shape.index(min(X.shape)) == 1:
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
                my_sampler = MyBatchSampler(range(len(train_dataset)), batch_size=self.batch_size, drop_last=True)
                train_loader = DataLoader(train_dataset, batch_sampler=my_sampler)
            
            # if self.proj_q:
            #     params = list(self.net_q.parameters()) + list(self.proj_q.parameters())
            #     print(f'=> Append projection head to encoder with dimension: {self.proj_dims}')
            # else:
            #     params = self.net_q.parameters()
            if self.proj_q:
                params = list(self.net_q.parameters()) + list(self.proj_q.parameters()) + list(self.net_k.parameters()) + list(self.proj_k.parameters())
                print(f'=> Append projection head to encoder with dimension: {self.proj_dims}')
            else:
                params = list(self.net_q.parameters()) + list(self.net_k.parameters())
                
            optimizer = torch.optim.AdamW(params, lr=self.lr)
            scheduler = self.get_scheduler(schedule, optimizer, epochs)
            if scheduler:
                print(f'=> Using scheduler: {schedule}')
                
            epoch_loss_list = []
            masks = mask_type.split('+') # e.g. 't+fb' -> ['t', 'fb']
            loss_func = id_contrastive_loss
                
            start_time = datetime.now()  
            for epoch in range(epochs):
                cum_loss = 0
                for x, y in tqdm(train_loader, desc=f'=> Epoch {epoch+1}', leave=False):
                    x = x.to(self.device)
                    # get encoder mask mode and transformed data
                    x1, mask1 = transform(x, opt=masks[0])
                    x2, mask2 = transform(x, opt=masks[1])
                    pid = y[:, 1]  # patient id
                    
                    # with torch.no_grad():
                    #     self._momentum_update_key_encoder()
                        
                    optimizer.zero_grad()
                    
                    q = self.net_q(x1, mask=mask1, pool=True)
                    if self.proj_q:
                        q = self.proj_q(q)
                    q = F.normalize(q, dim=1)
                    
                    k = self.net_k(x2, mask=mask2, pool=True)
                    if self.proj_k:
                        k = self.proj_k(k)
                    k = F.normalize(k, dim=1)
                    
                    # with torch.no_grad():
                    #     # shuffle BN
                    #     idx = torch.randperm(x2.size(0), device=x.device)
                    #     k = self.net_k(x2[idx], mask=mask2, pool=True)
                    #     if self.proj_k:
                    #         k = self.proj_k(k)
                    #     k = F.normalize(k, dim=1)
                    #     k = k[torch.argsort(idx)]
                    
                    loss = loss_func(q, k, pid)

                    loss.backward()
                    optimizer.step()
                    self.net_t.update_parameters(self.net_q)
                    self.net_f.update_parameters(self.net_k)

                    cum_loss += loss.item()
                    
                    # self._dequeue_and_enqueue(k, pid)

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
        
        
    # def _momentum_update_key_encoder(self):
    #     '''
    #     Momentum update of the key encoder
    #     '''
    #     for param_q, param_k in zip(
    #         self.net.parameters(), self.net_k.parameters()
    #     ):
    #         param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
            
    #     if self.proj_q:
    #         for param_q, param_k in zip(
    #             self.proj_q.parameters(), self.proj_k.parameters()
    #         ):
    #             param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)


    # def _dequeue_and_enqueue(self, keys, pid):
    #     '''
    #     update all queues
    #     '''
    #     # gather keys before updating queue
    #     batch_size = keys.shape[0]
        
    #     ptr = int(self.queue_ptr)
    #     assert self.queue_size % batch_size == 0  # for simplicity

    #     # replace the keys at ptr (dequeue and enqueue)
    #     self.queue[ptr : ptr + batch_size, ...] = keys
    #     self.id_queue[ptr : ptr + batch_size] = pid
    #     ptr = (ptr + batch_size) % self.queue_size  # move pointer

    #     self.queue_ptr[0] = ptr
        
        
    def get_scheduler(self, schedule, optimizer, epochs):
        if schedule == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 80], gamma=0.1)
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
        torch.save(self.net_t.state_dict(), fn)
    
    
    # def load(self, fn):
    #     '''Load the model from a file.
        
    #     Args:
    #         fn (str): filename.
    #     '''
    #     # state_dict = torch.load(fn, map_location=self.device)
    #     state_dict = torch.load(fn)
    #     self.net_t.load_state_dict(state_dict)
    
    
    
