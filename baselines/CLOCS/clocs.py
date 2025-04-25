"""
See the PMQ and CLOCS official repository for more details.
"""
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from flash.core.optimizers import LARS, LinearWarmupCosineAnnealingLR
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from encoder import ProjectionHead, TSEncoder
from utils import MyBatchSampler, shuffle_feature_label

class CLOCS:
    def __init__(
        self,
        input_dims=1,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        tau=0.1,
        device="cuda",
        multi_gpu=True,
    ):
        super().__init__()
        self.device = device
        
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        
        self.tau = tau
        
        self.multi_gpu = multi_gpu
        
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
      
        device = torch.device(device)
        if device == torch.device("cuda") and self.multi_gpu:
            # self.net_q = nn.DataParallel(self.net_q, device_ids=gpu_idx_list)
            self._net = nn.DataParallel(self._net)
        self._net.to(device)
        # stochastic weight averaging
        # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        # self.net = self._net
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        
        
    def fit(self, X, y, shuffle_function="random", epochs=None, batch_size=256, lr=1e-4, wd=1.5e-6, optim="adamw", schedule=None, logdir="", checkpoint=1, verbose=1):
        assert X.ndim == 4
        assert y.shape[1] == 3
        assert X.shape[-1] == 1
        
        # Shuffle the training set for contrastive learning pretraining.
        X, y = shuffle_feature_label(X, y, shuffle_function=shuffle_function, batch_size=batch_size)

        # we need patient id for patient-level contrasting and trial id for trial-level contrasting
        train_dataset = TensorDataset(
            torch.from_numpy(X).to(torch.float),
            torch.from_numpy(y).to(torch.long)
            )
        
        if shuffle_function == "random":
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            # Important!!! A customized batch_sampler to shuffle samples before each epoch. Check details in utils.py.
            my_sampler = MyBatchSampler(range(len(train_dataset)), batch_size=batch_size, drop_last=True)
            train_loader = DataLoader(train_dataset, batch_sampler=my_sampler)
        
        params = list(self._net.parameters())
        optimizer = self._get_optimizer(optim, params, lr, wd)
        scheduler = self._get_scheduler(schedule, optimizer, epochs, len(train_loader))
        
        epoch_loss_list = []
        start_time = datetime.now()           
        for epoch in range(epochs):
            cum_loss = 0
            for x, y in tqdm(train_loader, desc=f"=> Epoch {epoch+1}", leave=False):
                # count by iterations
                x = x.to(self.device)
                pid = y[:, 1]  # patient id

                optimizer.zero_grad()
                
                x1, x2 = x[:, 0], x[:, 1]  # two views
                z1 = self._net(x1)
                z2 = self._net(x2)
                
                z1 = F.normalize(z1, dim=-1)
                z2 = F.normalize(z2, dim=-1)
                
                loss1 = self.loss_fn(z1, z2, pid)
                loss2 = self.loss_fn(z2, z1, pid)
                loss = (loss1 + loss2) / 2
                
                loss.backward()
                optimizer.step()
                if schedule == "warmup":
                    scheduler.step()
                self.net.update_parameters(self._net)

                cum_loss += loss.item()
           
            cum_loss /= len(train_loader)
            epoch_loss_list.append(cum_loss)
            
            if schedule != "warmup":
                # if using reduceOnPlateau scheduler, pass the loss
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
            optimizer = torch.optim.AdamW(params, lr, weight_decay=wd)
        elif optim == "adam":
            optimizer = torch.optim.Adam(params, lr, weight_decay=wd)
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
    
    
    def loss_fn(self, z1, z2, id):
        """
        See CLOCS official repository for details.
        """
        id = id.cpu().detach().numpy()

        interest_matrix = np.equal.outer(id, id).astype(int) # B x B
        
        rows1, cols1 = np.where(np.triu(interest_matrix, 1))  # upper triangle same patient combs
        # rows2, cols2 = np.where(np.tril(interest_matrix, -1))  # down triangle same patient combs
        
        loss = 0
        eps = 1e-12
        
        sim_matrix = torch.mm(z1, z2.t()) # B x B
        argument = sim_matrix / self.tau
        sim_matrix_exp = torch.exp(argument)

        diag_elements = torch.diag(sim_matrix_exp)

        triu_sum = torch.sum(sim_matrix_exp, 1)  # add column
        # tril_sum = torch.sum(sim_matrix_exp, 0)  # add row

        loss = -torch.mean(torch.log((diag_elements + eps) / (torch.sum(sim_matrix_exp, 1) + eps)))
        # loss_diag2 = -torch.mean(torch.log((diag_elements + eps) / (torch.sum(sim_matrix_exp, 0) + eps)))
        # loss = loss_diag1 + loss_diag2
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
        return loss
    
    
    def save(self, fn):
        """Save the model to a file.
        
        Args:
            fn (str): filename.
        """
        torch.save(self.net.state_dict(), fn)
    
    
    def load(self, fn):
        raise NotImplementedError("Loading a model is not implemented yet.")
        
              
def cmsc_pretrain_split(x, y):
    length = x.shape[1]
    nleads = x.shape[-1]
    assert length % 2 == 0
    
    x = x.transpose(2, 0, 1).reshape(-1, 2, int(length/2), 1)
    y = np.tile(y, (nleads, 1))
    
    return x, y

def cmsc_finetune_split(x, y):
    length = x.shape[1]
    nleads = x.shape[-1]
    assert length % 2 == 0
    
    x = x.transpose(0, 2, 1).reshape(-1, length, 1)
    y = np.tile(y, (nleads, 1))
    
    return x, y