import os
from datetime import datetime
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from model.encoder import TSEncoder, ProjectionHead
from model.loss import hierarchical_contrastive_loss
from utils import shuffle_feature_label, MyBatchSampler, crop_overlap, take_topk_component

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
        
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        self.proj = ProjectionHead(input_dims=output_dims, output_dims=output_dims, hidden_dims=proj_dims) if proj_dims else None
        # self.proj_f = nn.Linear(output_dims, output_dims//2)
            
        device = torch.device(device)
        if device == torch.device('cuda') and self.multi_gpu:
            self._net = nn.DataParallel(self._net)
            if self.proj:
                self.proj = nn.DataParallel(self.proj)
                
        self._net.to(device)
        if self.proj:
            self.proj.to(device)
            
        # stochastic weight averaging
        # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        # self.net = self.net_q
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        
    
    def fit(self, X, y, shuffle_function='trial', mask='binomial', lamda=0.5, weights=[0.4, 0.3, 0.3], epochs=None, schedule=None, logdir='', checkpoint=1, verbose=1):
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
        
        if self.proj:
            params = list(self._net.parameters()) + list(self.proj.parameters())
            print(f'=> Append projection head to encoder with dimension: {self.proj_dims}')
        else:
            params = self._net.parameters()
            
        optimizer = torch.optim.AdamW(params, lr=self.lr)
        scheduler = self.get_scheduler(schedule, optimizer, epochs)
        if scheduler:
            print(f'=> Using scheduler: {schedule}')
            
        epoch_loss_list = []
            
        start_time = datetime.now()  
        for epoch in range(epochs):
            cum_loss = 0
            for x, y in tqdm(train_loader, desc=f'=> Epoch {epoch+1}', leave=False):
                x = x.to(self.device)
                pid = y[:, 1]  # patient id

                optimizer.zero_grad()
                
                x1, x2, crop_l = crop_overlap(x)
                
                out1 = self._net(x1, mask=mask)
                if self.proj:
                    out1 = self.proj(out1)
                crop1 = out1[:, -crop_l:, :]
                
                out2 = self._net(x2, mask=mask)
                if self.proj:
                    out2 = self.proj(out2)
                crop2 = out2[:, :crop_l, :]

                loss_t = hierarchical_contrastive_loss(crop1, crop2)
                
                # spec1 = take_topk_component(out1)
                # spec2 = take_topk_component(out2)
                # spec1 = self.proj_f(spec1)
                # spec2 = self.proj_f(spec2)
                spec1 = torch.fft.rfft(out1, dim=1, norm='ortho')
                spec2 = torch.fft.rfft(out2, dim=1, norm='ortho')
                
                loss_amp = hierarchical_contrastive_loss(spec1.abs(), spec2.abs())
                loss_phase = hierarchical_contrastive_loss(spec1.angle(), spec2.angle())

                loss_f = lamda * loss_amp + (1 - lamda) * loss_phase
                
                loss_p = hierarchical_contrastive_loss(crop1, crop2, pid)
                
                loss = weights[0] * loss_t + weights[1] * loss_f + weights[2] * loss_p
                
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self.net_q)

                cum_loss += loss.item()

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
        torch.save(self.net.state_dict(), fn)
    
    
    def load(self, fn):
        '''Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        # state_dict = torch.load(fn, map_location=self.device)
        state_dict = torch.load(fn)
        self.net.load_state_dict(state_dict)
    
