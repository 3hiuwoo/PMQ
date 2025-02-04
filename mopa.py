import torch
import os
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from model.encoder import TSEncoder
from model.loss import id_momentum_loss
from utils import shuffle_feature_label, MyBatchSampler, transform


class MOPA:
    '''A momentum contrastive learning model cross time, frequency, patient.
    
    Args:
        input_dims (int): The input dimension. For a uni-variate time series, this should be set to 1.
        output_dims (int): The representation dimension.
        hidden_dims (int): The hidden dimension of the encoder.
        depth (int): The number of hidden residual blocks in the encoder.
        device (str): The gpu used for training and inference.
        lr (float): The learning rate.
        batch_size (int): The batch size of samples.
        momentum (float): The momentum used for the key encoder.
        queue_size (int): The size of the queue.
        multi_gpu (bool): A flag to indicate whether using multiple gpus
        callback_func (Union[Callable, NoneType]): A callback function that would be called after each epoch.
    '''
    def __init__(
        self,
        input_dims=12,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=1e-4,
        batch_size=256,
        momentum=0.999,
        queue_size=65536,
        multi_gpu=True,
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
        
        self.net_q = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        self.net_k = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        
        for param_q, param_k in zip(
            self.net_q.parameters(), self.net_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        device = torch.device(device)
        if device == torch.device('cuda') and self.multi_gpu:
            # self.net_q = nn.DataParallel(self.net_q, device_ids=gpu_idx_list)
            self.net_q = nn.DataParallel(self.net_q)
            self.net_k = nn.DataParallel(self.net_k)
        self.net_q.to(device)
        self.net_k.to(device)
        # stochastic weight averaging
        # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        # self.net = self.net_q
        self.net = torch.optim.swa_utils.AveragedModel(self.net_q)
        self.net.update_parameters(self.net_q)

        # projection head append after encoder
        # self.proj_head = ProjectionHead(input_dims=self.output_dims, output_dims=2, hidden_dims=128).to(self.device)

        self.queue = torch.randn(queue_size, output_dims, device=device, requires_grad=False)
        self.queue = F.normalize(self.queue, dim=1)
        
        self.id_queue = torch.zeros(queue_size, dtype=torch.long, device=device, requires_grad=False)
        self.queue_ptr = torch.zeros(1, dtype=torch.long, device=device, requires_grad=False)
        
        
    def fit(self, X, y, shuffle_function='random', masks='o+fb', epochs=None, logdir='', checkpoint=1, verbose=True):
            ''' Training the MoPa model.
            
            Args:
                X (numpy.ndarray): The training data. It should have a shape of (n_samples, sample_timestamps, features).
                y (numpy.ndarray): The training labels. It should have a shape of (n_samples, 3). The three columns are the label, patient id, and trial id.
                shuffle_func (str): specify the shuffle function.
                masks (list): A list of masking functions applied (str).
                epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
                verbose (bool): Whether to print the training loss after each epoch.
                
            Returns:
                epoch_loss_list: a list containing the training losses on each epoch.
            '''
            assert X.ndim == 3
            assert y.shape[1] == 3
            # Shuffle the training set for contrastive learning pretraining.
            X, y = shuffle_feature_label(X, y, shuffle_function=shuffle_function, batch_size=self.batch_size)

            # we need patient id for patient-level contrasting and trial id for trial-level contrasting
            train_dataset = TensorDataset(
                torch.from_numpy(X).to(torch.float),
                torch.from_numpy(y).to(torch.long)
                )
            
            if shuffle_function == 'random':
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            else:
                # Important!!! A customized batch_sampler to shuffle samples before each epoch. Check details in utils.py.
                my_sampler = MyBatchSampler(range(len(train_dataset)), batch_size=self.batch_size, drop_last=True)
                train_loader = DataLoader(train_dataset, batch_sampler=my_sampler)
            
            optimizer = torch.optim.AdamW(self.net_q.parameters(), lr=self.lr)
            
            epoch_loss_list = []
            
            start_time = datetime.now() 
            masks = masks.split('+')          
            for epoch in range(epochs):
                cum_loss = 0
                for x, y in tqdm(train_loader, desc=f'=> Epoch {epoch+1}', leave=False):
                    x = x.to(self.device)
                    x1, mask1 = transform(x, opt=masks[0])
                    x2, mask2 = transform(x, opt=masks[1])
                    pid = y[:, 1]  # patient id
                    
                    with torch.no_grad():
                        self._momentum_update_key_encoder()
                        
                    optimizer.zero_grad()
                    
                    # do augmentation and compute representation
                    q = self.net_q(x1, mask=mask1, pool=True)
                    q = F.normalize(q, dim=1)
                    
                    with torch.no_grad():
                        # shuffle BN
                        idx = torch.randperm(x2.size(0), device=x.device)
                        k = self.net_k(x2[idx], mask=mask2, pool=True)
                        k = F.normalize(k, dim=1)
                        k = k[torch.argsort(idx)]

                    # loss calculation
                    loss = id_momentum_loss(q, k, self.queue.clone().detach(), pid, self.id_queue.clone().detach())

                    loss.backward()
                    optimizer.step()
                    self.net.update_parameters(self.net_q)

                    cum_loss += loss.item()
                    
                    self._dequeue_and_enqueue(k, pid)
            
                cum_loss /= len(train_loader)
                epoch_loss_list.append(cum_loss)
                
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


    def _dequeue_and_enqueue(self, keys, pid):
        # gather keys before updating queue
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr : ptr + batch_size, ...] = keys
        self.id_queue[ptr : ptr + batch_size] = pid
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr
        
   
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
    
    
