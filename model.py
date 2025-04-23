"""
PMQ model with queue and without queue.
"""
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from flash.core.optimizers import LARS, LinearWarmupCosineAnnealingLR
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from encoder import MLP, TSEncoder
from utils import MyBatchSampler, shuffle_feature_label

class PMQ:
    """ PMQ with queue.
    Args:
        input_dims (int): The input dimension. For a uni-variate time series, this should be set to 1.
        output_dims (int): The representation dimension.
        hidden_dims (int): The dimension of input projector.
        depth (int): The number of hidden residual blocks in the encoder.
        pool (str): The pooling method for the representation.
        mask_t (float): The temporal masking probability.
        mask_f (float): The frequency masking ratio.
        momentum (float): The momentum update parameter.
        tau (float): The temperature parameter.
        alpha (float): The scaling factor of positive pairs in multi-similarity loss.
        beta (float): The scaling factor of negative pairs in multi-similarity loss.
        thresh (float): The bias for pairs in multi-similarity loss.
        margin (float): The margin for mining pairs in multi-similarity loss.
        queue_size (int): The size of the queue.
        use_id (bool): A flag to indicate whether using patient id for patient-level contrastive learning.
        loss_func (str): The loss function used for training. It can be "ms" or "nce".
        device (str): The gpu used for training and inference.
        multi_gpu (bool): A flag to indicate whether using multiple gpus
    """
    def __init__(
        self,
        input_dims=12,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        pool="avg",
        mask_t=0.5,
        mask_f=0.1,
        momentum=0.99,
        tau=1.0,
        alpha=2,
        beta=50,
        thresh=1.0,
        margin=0.1,
        queue_size=16384,
        use_id=True,
        loss_func="ms",
        device="cuda",
        multi_gpu=False
    ):
        super().__init__()
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.pool = pool
        self.mask_t = mask_t
        self.mask_f = mask_f
        self.device = device
        self.multi_gpu = multi_gpu
        
        if not use_id:
            print("=> !!! Training without patient IDs !!!")
        self.use_id = use_id
        self.loss_func = loss_func
        
        self.momentum = momentum
        self.tau = tau
        
        # following 4 parameters are only used for multi-similarity loss
        self.alpha = alpha
        self.beta = beta
        self.thresh = thresh
        self.margin = margin
        
        self.queue_size = queue_size
        
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth, mask_t=mask_t, mask_f=mask_f, pool=pool)
        self._proj = MLP(input_dims=output_dims, output_dims=output_dims, nlayers=2, hidden_dims=output_dims)
        self._pred = MLP(input_dims=output_dims, output_dims=output_dims, nlayers=1, hidden_dims=output_dims)
        
        self.momentum_net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth, mask_t=mask_t, mask_f=mask_f, pool=pool)
        self.momentum_proj = MLP(input_dims=output_dims, output_dims=output_dims, hidden_dims=output_dims)
        
        self._momentum_init()
                
        device = torch.device(device)
        if device == torch.device("cuda") and self.multi_gpu:
            self._net_t = nn.DataParallel(self._net_t)
            self._net_f = nn.DataParallel(self._net_f)
            self.momentum_net_t = nn.DataParallel(self.momentum_net_t)
            self.momentum_net_f = nn.DataParallel(self.momentum_net_f)
            self.proj = nn.DataParallel(self.proj)
            self.momentum_proj = nn.DataParallel(self.momentum_proj)
                
        self._net.to(device)
        self._proj.to(device)
        self._pred.to(device)
        self.momentum_net.to(device)
        self.momentum_proj.to(device)
        
        # Use stochastic weight averaging
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        self.proj = torch.optim.swa_utils.AveragedModel(self._proj)
        self.proj.update_parameters(self._proj)
        self.pred = torch.optim.swa_utils.AveragedModel(self._pred)
        self.pred.update_parameters(self._pred)
        
        # patient representation queue
        self.queue = torch.randn(queue_size, output_dims, device=device, requires_grad=False)
        self.queue = F.normalize(self.queue, dim=1)
        # patient ID queue
        self.id_queue = torch.zeros(queue_size, dtype=torch.long, device=device, requires_grad=False) if use_id else None
        # shared queue pointer
        self.queue_ptr = torch.zeros(1, dtype=torch.long, device=device, requires_grad=False)
    
           
    def fit(self, X, y, shuffle_function="trial", epochs=None, batch_size=256, lr=1e-4, wd=1.5e-6, optim="adamw", schedule=None, logdir="", checkpoint=1, verbose=1):
        """ Training the model.
        Args:
            X (numpy.ndarray): The training data with shape of (n_samples, sample_timestamps, features) or (n_samples, 2, sample_timestamps, features).
            y (numpy.ndarray): The training labels. It should have a shape of (n_samples, 3). The three columns are the label, patient id, and trial id.
            shuffle_function (str): specify the shuffle method (random/trial).
            mask_t (int): Temporal masking probability.
            mask_f (int): Frequency masking ratio.
            epochs (Union[int, NoneType]): The number of epochs..
            batch_size (int): The batch size.
            lr (float): The learning rate.
            wd (float): The weight decay.
            optim (str): The optimizer used for training.
            schedule (str): The learning rate scheduler.
            logdir (str): The directory to save the model.
            checkpoint (int): The number of epochs to save the model.
            verbose (int): >0 to print the training loss after each epoch.
        Returns:
            epoch_loss_list (list): a list containing the training losses on each epoch.
        """
        assert y.shape[1] == 3
        print("=> Number of dimension of training data:", X.ndim)
        
        if shuffle_function == "trial": # shuffle the data in trial level
            X, y = shuffle_feature_label(X, y, shuffle_function=shuffle_function, batch_size=batch_size)

        train_dataset = TensorDataset(
            torch.from_numpy(X).to(torch.float),
            torch.from_numpy(y).to(torch.long)
            )
        
        if shuffle_function == "random":
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            print("=> Shuffle data by trial")
            my_sampler = MyBatchSampler(range(len(train_dataset)), batch_size=batch_size, drop_last=True)
            train_loader = DataLoader(train_dataset, batch_sampler=my_sampler)
        
        params = list(self._net.parameters()) + list(self._proj.parameters()) + list(self._pred.parameters())
        optimizer = self._get_optimizer(optim, params, lr, wd)
        scheduler = self._get_scheduler(schedule, optimizer, epochs, len(train_loader))
                  
        epoch_loss_list = []
            
        start_time = datetime.now()  
        for epoch in range(epochs):
            cum_loss = 0
            for x, y in tqdm(train_loader, desc=f"=> Epoch {epoch+1}", leave=False):
                x = x.to(self.device)
                pid = y[:, 1] if self.use_id else None

                self._momentum_update()
                    
                optimizer.zero_grad()
                
                # check if the data is segmented into neighbors
                if x.ndim == 4:
                    x1 = x[:, 0, ...]
                    x2 = x[:, 1, ...]
                else:
                    x1 = x
                    x2 = x
                
                q = self._net(x1)
                q = self._proj(q)
                q = self._pred(q)
                q = F.normalize(q, dim=-1)
                
                with torch.no_grad():
                    k = self.momentum_net(x2)
                    k = self.momentum_proj(k)
                k = F.normalize(k, dim=-1)
                
                loss = self.loss_fn(q, k, pid)
                
                loss.backward()
                optimizer.step()
                
                # warmup by step not epoch
                if schedule == "warmup":
                    scheduler.step()
                self._update_swa() # stochastic weight averaging

                cum_loss += loss.item()
                
                self._update_queue(k, pid)

            cum_loss /= len(train_loader)
            epoch_loss_list.append(cum_loss)
            
            if schedule != "warmup":
                if schedule == "plateau":
                    scheduler.step(cum_loss)
                elif scheduler:
                    scheduler.step()
            
            if verbose:
                print(f"=> Epoch {epoch+1}: loss: {cum_loss}")
                
            if (epoch+1) % checkpoint == 0:
                self.save(os.path.join(logdir, f"pretrain_{epoch+1}.pth"))
                
        end_time = datetime.now()
        print(f"=> Training finished in {end_time - start_time}")
            
        return epoch_loss_list
        
    
    def _momentum_init(self):
        """
        Initialize the momentum encoder and projector with the same weights as the original encoder and projector.
        """
        for param_q, param_k in zip(self._net.parameters(), self.momentum_net.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self._proj.parameters(), self.momentum_proj.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
            
    def _update_swa(self):
        """
        update SWA models
        """
        self.net.update_parameters(self._net)
        self.proj.update_parameters(self._proj)
        self.pred.update_parameters(self._pred)
        
    
    def _momentum_update(self):
        """
        perform momentum update
        """
        with torch.no_grad():
            for param_q, param_k in zip(
                self.net.parameters(), self.momentum_net.parameters()
            ):
                param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
                
            for param_q, param_k in zip(
                self.proj.parameters(), self.momentum_proj.parameters()
            ):
                param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
            
                
    def _update_queue(self, k, pid=None):
        """ perform enqueue and dequeue
        Args:
            k (torch.Tensor): representations from momentum projector
            pid (torch.Tensor, optional): patient id
        """
        # gather keys before updating queue
        batch_size = k.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr : ptr + batch_size, ...] = k
        
        if pid is not None:
            self.id_queue[ptr : ptr + batch_size] = pid
            
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr
        
    
    def _get_optimizer(self, optim, params, lr, wd):
        """ get optimizer
        Args:
            optim (str): optimizer name
            params (list): list of parameters
            lr (float): learning rate
            wd (float): weight decay
        Returns:
            optimizer(torch.optim.Optimizer): optimizer
        """
        if optim == "adamw":
            optimizer = torch.optim.AdamW(params, lr)
        elif optim == "lars":
            optimizer = LARS(params, lr, weight_decay=wd)
        else:
            raise ValueError(f"{optim} is not supported")
        return optimizer
    
    
    def _get_scheduler(self, schedule, optimizer, epochs, iters):
        """ get scheduler
        Args:
            schedule (str): scheduler name
            optimizer (torch.optim.Optimizer): optimizer
            epochs (int): number of epochs
            iters (int): number of iterations per epoch, used by warmup scheduler only.
        Returns:
            scheduler(torch.optim.lr_scheduler._LRScheduler): scheduler
        """
        if schedule == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
        elif schedule == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        elif schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif schedule == "exp":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif schedule == "warmup":
            steps = epochs * iters
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, int(0.05*steps), steps)
        else:
            scheduler = None
            
        return scheduler
    
    
    def save(self, fn):
        """Save the model to a file.
        
        Args:
            fn (str): filename.
        """
        torch.save(self.net.state_dict(), fn)
    
    
    def load(self, fn):
        raise NotImplementedError
    
    
    def loss_fn(self, q, k, id=None):
        """ compute the contrastive loss
        Args:
            q (torch.Tensor): query representations
            k (torch.Tensor): key representations
            id (torch.Tensor, optional): patient id, set None to use infoNCE loss.
        Returns:
            loss (torch.Tensor): loss
        """
        if id is None:
            return self.infoNCE_loss(q, k)
        elif self.loss_func == "ms":
            return self.ms_loss(q, k, id)
        elif self.loss_func == "nce":
            return self.patient_infoNCE_loss(q, k, id)
        else:
            raise ValueError(f"{self.loss_func} is not supported")
    
    
    def infoNCE_loss(self, q, k):
        """ compute the infoNCE loss
        Args:
            q (torch.Tensor): query representations
            k (torch.Tensor): key representations
        Returns:
            loss (torch.Tensor): loss
        """
        N, C = q.shape
        l_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).squeeze(-1)  # positive logits: Nx1
        l_neg = torch.mm(q, self.queue.t())  # negative logits: NxK
        logits = torch.cat([l_pos, l_neg], dim=1)  # logits: Nx(1+K)
        labels = torch.zeros(N, dtype=torch.long, device=q.device)  # positives are the 0-th
        loss = F.cross_entropy(logits / self.tau, labels)
        return 2 * self.tau * loss
        
        
    def patient_infoNCE_loss(self, q, k, id):
        """ compute the multi-infoNCE loss
        Args:
            q (torch.Tensor): query representations
            k (torch.Tensor): key representations
            id (torch.Tensor): patient id
        Returns:
            loss (torch.Tensor): loss
        """
        id = id.cpu().detach().numpy()
        id_queue = self.id_queue.clone().detach().cpu().numpy()
        queue = self.queue.clone().detach()

        batch_interest_matrix = np.equal.outer(id, id).astype(int) # B x B
        queue_interest_matrix = np.equal.outer(id, id_queue).astype(int) # B x K
        interest_matrix = np.concatenate((batch_interest_matrix, queue_interest_matrix), axis=1) # B x (B+K)
        
        # only consider upper diagnoal where the queue is taken into account
        rows1, cols1 = np.where(np.triu(interest_matrix, 1))  # upper triangle same patient combs
        # rows2, cols2 = np.where(np.tril(interest_matrix, -1))  # down triangle same patient combs
        
        eps = 1e-12
        batch_sim_matrix = torch.mm(q, k.t()) # B x B
        queue_sim_matrix = torch.mm(q, queue.t()) # B x K
        sim_matrix = torch.cat((batch_sim_matrix, queue_sim_matrix), dim=1) # B x (B+K)
        argument = sim_matrix / self.tau
        sim_matrix_exp = torch.exp(argument)

        diag_elements = torch.diag(sim_matrix_exp)

        triu_sum = torch.sum(sim_matrix_exp, 1)  # add column
        # tril_sum = torch.sum(sim_matrix_exp, 0)  # add row

        loss = -torch.mean(torch.log((diag_elements + eps) / (torch.sum(sim_matrix_exp, 1) + eps)))
        if len(rows1) > 0:
            triu_elements = sim_matrix_exp[rows1, cols1]  # row and column for upper triangle same patient combinations
            loss_triu = -torch.mean(torch.log((triu_elements + eps) / (triu_sum[rows1] + eps)))
            loss += loss_triu 
            loss /= 2
            
        return 2 * self.tau * loss
    
    
    def ms_loss(self, q, k, id):
        """ compute the multi-similarity loss
        Args:
            q (torch.Tensor): query representations
            k (torch.Tensor): key representations
            id (torch.Tensor): patient id
        Returns:
            loss (torch.Tensor): loss
        """
        id = id.cpu().detach().numpy()
        id_queue = self.id_queue.clone().detach().cpu().numpy()
        queue = self.queue.clone().detach()

        batch_interest_matrix = np.equal.outer(id, id).astype(int)
        queue_interest_matrix = np.equal.outer(id, id_queue).astype(int) # B x K
        interest_matrix = np.concatenate((batch_interest_matrix, queue_interest_matrix), axis=1) # B x (B+K)

        batch_sim_matrix = torch.mm(q, k.t()) # B x B
        queue_sim_matrix = torch.mm(q, queue.t()) # B x K
        sim_matrix = torch.cat((batch_sim_matrix, queue_sim_matrix), dim=1) # B x (B+K)
        sim_matrix /= self.tau
        # sim_matrix_exp = torch.exp(argument)
        
        eps = 1e-5
        # loss = []
        pos_pairs = []
        neg_pairs = []
        for i in range(sim_matrix.shape[0]):
            pos_pair_ = sim_matrix[i][interest_matrix[i] == 1] # positive pairs
            pos_pair_ = pos_pair_[pos_pair_ < 1 - eps]
            neg_pair_ = sim_matrix[i][interest_matrix[i] == 0] # negative pairs
            
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)] # cherry-pick positive pairs
            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)] # cherry-pick negative pairs
            
            if len(pos_pair) == 0 or len(neg_pair) == 0:
                continue
            
            pos_pairs.append(pos_pair)
            neg_pairs.append(neg_pair)
            
            # pos_loss = 1.0 / self.alpha * torch.log(
            #     1 + torch.sum(torch.exp(-self.alpha * (pos_pair - self.thresh))) # positive pairs
            # )
            # neg_loss = 1.0 / self.beta * torch.log(
            #     1 + torch.sum(torch.exp(self.beta * (neg_pair - self.thresh))) # negative pairs   
            # )
            # loss.append(pos_loss + neg_loss)
        
        max_pos = max(len(pos) for pos in pos_pairs)
        max_neg = max(len(neg) for neg in neg_pairs)
        
        pad_pos_pair = torch.stack([F.pad(pos, (0, max_pos - len(pos)), value=0) for pos in pos_pairs])
        pad_neg_pair = torch.stack([F.pad(neg, (0, max_neg - len(neg)), value=0) for neg in neg_pairs])
        
        pos_loss = torch.mean(1.0 / self.alpha * torch.log(
            1 + torch.sum(torch.exp(-self.alpha * (pad_pos_pair - self.thresh)), dim=1) # positive pairs
        ))
        neg_loss = torch.mean(1.0 / self.beta * torch.log(
            1 + torch.sum(torch.exp(self.beta * (pad_neg_pair - self.thresh)), dim=1) # negative pairs
        ))
        
        return pos_loss + neg_loss
    
        # if len(loss) == 0:
        #     print("=> No loss, return 0")
        #     return torch.tensor(0.0, device=q.device)
        # else:
        #     loss = sum(loss) / sim_matrix.shape[0]
        #     return loss
        
        
        
class PMB:
    """ PMQ without queue.
    Args:
        input_dims (int): The input dimension. For a uni-variate time series, this should be set to 1.
        output_dims (int): The representation dimension.
        hidden_dims (int): The dimension of input projector.
        depth (int): The number of hidden residual blocks in the encoder.
        pool (str): The pooling method for the representation.
        mask_t (float): The temporal masking probability.
        mask_f (float): The frequency masking ratio.
        momentum (float): The momentum update parameter.
        tau (float): The temperature parameter.
        alpha (float): The scaling factor of positive pairs in multi-similarity loss.
        beta (float): The scaling factor of negative pairs in multi-similarity loss.
        thresh (float): The bias for pairs in multi-similarity loss.
        margin (float): The margin for mining pairs in multi-similarity loss.
        loss_func (str): The loss function used for training. It can be "ms" or "nce", since PMB has no queue, one can add s to the loss_func allowing computing the loss symmetrically.
        use_id (bool): A flag to indicate whether using patient id for patient-level contrastive learning.
        device (str): The gpu used for training and inference.
        multi_gpu (bool): A flag to indicate whether using multiple gpus
    """
    def __init__(
        self,
        input_dims=12,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        pool="avg",
        mask_t=0.5,
        mask_f=0.1,
        momentum=0.999,
        tau=0.1,
        alpha=2,
        beta=50,
        thresh=1.0,
        margin=0.1,
        use_id=True,
        loss_func="ms",
        device="cuda",
        multi_gpu=False
    ):
        super().__init__()
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.pool = pool
        self.mask_t = mask_t
        self.mask_f = mask_f
        self.device = device
        self.multi_gpu = multi_gpu
        
        if not use_id:
            print("=> !!! Training without patient IDs !!!")
        self.use_id = use_id
        self.loss_func = loss_func
        
        self.momentum = momentum
        self.tau = tau
        
        # following 4 parameters are only used for multi-similarity loss
        self.alpha = alpha
        self.beta = beta
        self.thresh = thresh
        self.margin = margin
        
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth, mask_t=mask_t, mask_f=mask_f, pool=pool)
        self._proj = MLP(input_dims=output_dims, output_dims=output_dims, nlayers=2, hidden_dims=output_dims)
        self._pred = MLP(input_dims=output_dims, output_dims=output_dims, nlayers=1, hidden_dims=output_dims)
        
        self.momentum_net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth, mask_t=mask_t, mask_f=mask_f, pool=pool)
        self.momentum_proj = MLP(input_dims=output_dims, output_dims=output_dims, hidden_dims=output_dims)
        
        self._momentum_init()
                
        device = torch.device(device)
        if device == torch.device("cuda") and self.multi_gpu:
            self._net_t = nn.DataParallel(self._net_t)
            self._net_f = nn.DataParallel(self._net_f)
            self.momentum_net_t = nn.DataParallel(self.momentum_net_t)
            self.momentum_net_f = nn.DataParallel(self.momentum_net_f)
            self.proj = nn.DataParallel(self.proj)
            self.momentum_proj = nn.DataParallel(self.momentum_proj)
                
        self._net.to(device)
        self._proj.to(device)
        self._pred.to(device)
        self.momentum_net.to(device)
        self.momentum_proj.to(device)
        
        # Use stochastic weight averaging
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        self.proj = torch.optim.swa_utils.AveragedModel(self._proj)
        self.proj.update_parameters(self._proj)
        self.pred = torch.optim.swa_utils.AveragedModel(self._pred)
        self.pred.update_parameters(self._pred)
        

    def fit(self, X, y, shuffle_function="trial", epochs=None, batch_size=256, lr=1e-4, wd=1.5e-6, optim="adamw", schedule=None, logdir="", checkpoint=1, verbose=1):
        """ Training the model.
        Args:
            X (numpy.ndarray): The training data with shape of (n_samples, sample_timestamps, features) or (n_samples, 2, sample_timestamps, features).
            y (numpy.ndarray): The training labels. It should have a shape of (n_samples, 3). The three columns are the label, patient id, and trial id.
            shuffle_function (str): specify the shuffle method (random/trial).
            mask_t (int): Temporal masking probability.
            mask_f (int): Frequency masking ratio.
            epochs (Union[int, NoneType]): The number of epochs..
            batch_size (int): The batch size.
            lr (float): The learning rate.
            wd (float): The weight decay.
            optim (str): The optimizer used for training.
            schedule (str): The learning rate scheduler.
            logdir (str): The directory to save the model.
            checkpoint (int): The number of epochs to save the model.
            verbose (int): >0 to print the training loss after each epoch.
        Returns:
            epoch_loss_list (list): a list containing the training losses on each epoch.
        """
        assert y.shape[1] == 3
        print("=> Number of dimension of training data:", X.ndim)
        
        if shuffle_function == "trial":
            X, y = shuffle_feature_label(X, y, shuffle_function=shuffle_function, batch_size=batch_size)

        # we need patient id for patient-level contrasting and trial id for trial-level contrasting
        train_dataset = TensorDataset(
            torch.from_numpy(X).to(torch.float),
            torch.from_numpy(y).to(torch.long)
            )
        
        if shuffle_function == "random":
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            print("=> Shuffle data by trial")
            my_sampler = MyBatchSampler(range(len(train_dataset)), batch_size=batch_size, drop_last=True)
            train_loader = DataLoader(train_dataset, batch_sampler=my_sampler)
        
        params = list(self._net.parameters()) + list(self._proj.parameters()) + list(self._pred.parameters())
        optimizer = self._get_optimizer(optim, params, lr, wd)
        scheduler = self._get_scheduler(schedule, optimizer, epochs, len(train_loader))
                  
        epoch_loss_list = []
            
        start_time = datetime.now()  
        for epoch in range(epochs):
            cum_loss = 0
            for x, y in tqdm(train_loader, desc=f"=> Epoch {epoch+1}", leave=False):
                x = x.to(self.device)
                pid = y[:, 1] if self.use_id else None

                self._momentum_update()
                    
                optimizer.zero_grad()
                
                # check if the data is segmented into neighbors
                if x.ndim == 4:
                    x1 = x[:, 0, ...]
                    x2 = x[:, 1, ...]
                else:
                    x1 = x
                    x2 = x
                
                q1 = self._net(x1)
                q1 = self._proj(q1)
                q1 = self._pred(q1)
                
                q2 = self._net(x2)
                q2 = self._proj(q2)
                q2 = self._pred(q2)
                
                q1 = F.normalize(q1, dim=-1)
                q2 = F.normalize(q2, dim=-1)
                
                with torch.no_grad():
                    k1 = self.momentum_net(x1)
                    k1 = self.momentum_proj(k1)
                    
                    k2 = self.momentum_net(x2)
                    k2 = self.momentum_proj(k2)
                
                k1 = F.normalize(k1, dim=-1)
                k2 = F.normalize(k2, dim=-1)
                
                loss = self.loss_fn(q1, k2, pid)
                # the last character of loss_func is used to indicate whether to compute loss symmetrically
                if self.loss_func[-1] == "s":
                    loss += self.loss_fn(q2, k1, pid)
                
                loss.backward()
                optimizer.step()
                
                # warmup by step not epoch
                if schedule == "warmup":
                    scheduler.step()
                self._update_swa() # stochastic weight averaging

                cum_loss += loss.item()

            cum_loss /= len(train_loader)
            epoch_loss_list.append(cum_loss)
            
            if schedule != "warmup":
                if schedule == "plateau":
                    scheduler.step(cum_loss)
                elif scheduler:
                    scheduler.step()
            
            if verbose:
                print(f"=> Epoch {epoch+1}: loss: {cum_loss}")
                
            if (epoch+1) % checkpoint == 0:
                self.save(os.path.join(logdir, f"pretrain_{epoch+1}.pth"))
                
        end_time = datetime.now()
        print(f"=> Training finished in {end_time - start_time}")
            
        return epoch_loss_list
        
    
    def _momentum_init(self):
        """
        Initialize the momentum encoder and projector with the same weights as the original encoder and projector.
        """
        for param_q, param_k in zip(self._net.parameters(), self.momentum_net.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self._proj.parameters(), self.momentum_proj.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
            
    def _update_swa(self):
        """
        update SWA models
        """
        self.net.update_parameters(self._net)
        self.proj.update_parameters(self._proj)
        self.pred.update_parameters(self._pred)
        
    
    def _momentum_update(self):
        """
        perform momentum update
        """
        with torch.no_grad():
            for param_q, param_k in zip(
                self.net.parameters(), self.momentum_net.parameters()
            ):
                param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
                
            for param_q, param_k in zip(
                self.proj.parameters(), self.momentum_proj.parameters()
            ):
                param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
            

    def _get_optimizer(self, optim, params, lr, wd):
        """ get optimizer
        Args:
            optim (str): optimizer name
            params (list): list of parameters
            lr (float): learning rate
            wd (float): weight decay
        Returns:
            optimizer(torch.optim.Optimizer): optimizer
        """
        if optim == "adamw":
            optimizer = torch.optim.AdamW(params, lr)
        elif optim == "lars":
            optimizer = LARS(params, lr, weight_decay=wd)
        else:
            raise ValueError(f"{optim} is not supported")
        return optimizer
    
    
    def _get_scheduler(self, schedule, optimizer, epochs, iters):
        """ get scheduler
        Args:
            schedule (str): scheduler name
            optimizer (torch.optim.Optimizer): optimizer
            epochs (int): number of epochs
            iters (int): number of iterations per epoch, used by warmup scheduler only.
        Returns:
            scheduler(torch.optim.lr_scheduler._LRScheduler): scheduler
        """
        if schedule == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)
        elif schedule == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        elif schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif schedule == "exp":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        elif schedule == "warmup":
            steps = epochs * iters
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, int(0.05*steps), steps)
        else:
            scheduler = None
            
        return scheduler
    
    
    def save(self, fn):
        """Save the model to a file.
        
        Args:
            fn (str): filename.
        """
        torch.save(self.net.state_dict(), fn)
    
    
    def load(self, fn):
        raise NotImplementedError
    
    
    def loss_fn(self, q, k, id=None):
        """ compute the patient infoNCE/infoNCE loss
        Args:
            q (torch.Tensor): query representations
            k (torch.Tensor): key representations
            id (torch.Tensor, optional): patient id, set None to use infoNCE loss.
        Returns:
            loss (torch.Tensor): loss
        """
        if id is None:
            return self.infoNCE_loss(q, k)
        # the last character of loss_func is used to indicate whether to compute loss symmetrically
        elif self.loss_func[:-1] == "ms":
            return self.ms_loss(q, k, id)
        elif self.loss_func[:-1] == "nce":
            return self.patient_infoNCE_loss(q, k, id)
        else:
            raise ValueError(f"{self.loss_func} is not supported")
    
    
    def infoNCE_loss(self, q, k):
        """ compute the infoNCE loss
        Args:
            q (torch.Tensor): query representations
            k (torch.Tensor): key representations
        Returns:
            loss (torch.Tensor): loss
        """
        logits = torch.mm(q, k.t())  # [N, N] pairs
        labels = torch.arange(logits.size(0), device=logits.device)  # positives are in diagonal
        loss = F.cross_entropy(logits / self.tau, labels)
        return 2 * self.tau * loss
    
      
    def patient_infoNCE_loss(self, q, k, id):
        """ compute the patient infoNCE loss
        Args:
            q (torch.Tensor): query representations
            k (torch.Tensor): key representations
            id (torch.Tensor): patient id
        Returns:
            loss (torch.Tensor): loss
        """
        id = id.cpu().detach().numpy()

        interest_matrix = np.equal.outer(id, id).astype(int) # B x B
        
        rows1, cols1 = np.where(np.triu(interest_matrix, 1))  # upper triangle same patient combs
        # rows2, cols2 = np.where(np.tril(interest_matrix, -1))  # down triangle same patient combs
        
        loss = 0
        eps = 1e-12
        
        sim_matrix = torch.mm(q, k.t()) # B x B
        argument = sim_matrix / self.tau
        sim_matrix_exp = torch.exp(argument)

        diag_elements = torch.diag(sim_matrix_exp)

        triu_sum = torch.sum(sim_matrix_exp, 1)  # add column
        # tril_sum = torch.sum(sim_matrix_exp, 0)  # add row

        loss_diag1 = -torch.mean(torch.log((diag_elements + eps) / (torch.sum(sim_matrix_exp, 1) + eps)))
        # loss_diag2 = -torch.mean(torch.log((diag_elements + eps) / (torch.sum(sim_matrix_exp, 0) + eps)))
        loss = loss_diag1 # + loss_diag2
        loss_term = 1
        
        # upper triangle same patient combs exist
        if len(rows1) > 0:
            triu_elements = sim_matrix_exp[rows1, cols1]  # row and column for upper triangle same patient combinations
            loss_triu = -torch.mean(torch.log((triu_elements + eps) / (triu_sum[rows1] + eps)))
            loss += loss_triu  # technicalneed to add 1 more term for symmetry
            loss_term += 1
        
        # down triangle same patient combs exist
        # if len(rows2) > 0:
        #     eps = 1e-12
        #     tril_elements = sim_matrix_exp[rows2, cols2]
        #     loss_tril = -torch.mean(torch.log((tril_elements + eps) / (tril_sum[cols2] + eps)))
        #     loss += loss_tril
        #     loss_term += 1
            
        loss /= loss_term
        return 2 * self.tau * loss
    
    
    def ms_loss(self, q, k, id):
        """ compute the multi-similarity loss
        Args:
            q (torch.Tensor): query representations
            k (torch.Tensor): key representations
            id (torch.Tensor): patient id
        Returns:
            loss (torch.Tensor): loss
        """
        id = id.cpu().detach().numpy()

        interest_matrix = np.equal.outer(id, id).astype(int) # B x B

        sim_matrix = torch.mm(q, k.t()) # B x B
        sim_matrix /= self.tau
        # sim_matrix_exp = torch.exp(argument)
        
        eps = 1e-5
        loss = []
        # pos_pairs = []
        # neg_pairs = []
        for i in range(sim_matrix.shape[0]):
            pos_pair_ = sim_matrix[i][interest_matrix[i] == 1] # positive pairs
            pos_pair_ = pos_pair_[pos_pair_ < 1 - eps]
            neg_pair_ = sim_matrix[i][interest_matrix[i] == 0] # negative pairs
            
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)] # cherry-pick positive pairs
            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)] # cherry-pick negative pairs
            
            if len(pos_pair) == 0 or len(neg_pair) == 0:
                continue
            
            # pos_pairs.append(pos_pair)
            # neg_pairs.append(neg_pair)
            
            pos_loss = 1.0 / self.alpha * torch.log(
                1 + torch.sum(torch.exp(-self.alpha * (pos_pair - self.thresh))) # positive pairs
            )
            neg_loss = 1.0 / self.beta * torch.log(
                1 + torch.sum(torch.exp(self.beta * (neg_pair - self.thresh))) # negative pairs   
            )
            loss.append(pos_loss + neg_loss)
            
        # max_pos = max(len(pos) for pos in pos_pairs)
        # max_neg = max(len(neg) for neg in neg_pairs)
        
        # pad_pos_pair = torch.stack([F.pad(pos, (0, max_pos - len(pos)), value=0) for pos in pos_pairs])
        # pad_neg_pair = torch.stack([F.pad(neg, (0, max_neg - len(neg)), value=0) for neg in neg_pairs])
        
        # pos_loss = torch.mean(1.0 / self.alpha * torch.log(
        #     1 + torch.sum(torch.exp(-self.alpha * (pad_pos_pair - self.thresh)), dim=1) # positive pairs
        # ))
        # neg_loss = torch.mean(1.0 / self.beta * torch.log(
        #     1 + torch.sum(torch.exp(self.beta * (pad_neg_pair - self.thresh)), dim=1) # negative pairs
        # ))
        
        # return pos_loss + neg_loss
    
        if len(loss) == 0:
            print("=> No loss, return 0")
            return torch.tensor(0.0, device=q.device)
        else:
            loss = sum(loss) / sim_matrix.shape[0]
            return loss