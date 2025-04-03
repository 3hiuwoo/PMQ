import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from flash.core.optimizers import LARS, LinearWarmupCosineAnnealingLR
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from data import load_data
from encoder import MLP, TSEncoder


class ECGDataset(Dataset):
    def __init__(self, root="/root/autodl-tmp/dataset", name="chapman", length=300, overlap=0., norm=True, neighbor=False):
        """ PyTorch Dataset for ECG data.

        Args:
            root (str): Root directory containing the dataset.
            name (str): Dataset name (e.g., "chapman", "ptb", etc.).
            length (int): Segment length.
            overlap (float): Overlap ratio for splitting trials into segments.
            norm (bool): Whether to normalize the data.
            neighbor (bool): Whether to split the data into two halves.
        """
        # Load the data using the provided load_data function
        X_train, _, _, y_train, _, _ = load_data(root=root, name=name, length=length, overlap=overlap, norm=norm, neighbor=neighbor)
        
        self.X_train = X_train
        self.y_train = y_train

        # Group segments by trial ID
        self.patient_segemtns = {}
        for i, label in enumerate(y_train):
            pid = label[1]  # Assuming the third column is the trial ID
            if pid not in self.patient_segemtns:
                self.patient_segemtns[pid] = []
            self.patient_segemtns[pid].append(self.X_train[i])

        # Convert trial segments to numpy arrays for efficient indexing
        for pid in self.patient_segemtns:
            self.patient_segemtns[pid] = np.array(self.patient_segemtns[pid])

        self.pids = list(self.patient_segemtns.keys())


    def __len__(self):
        """
        Return the number of trials.
        """
        return len(self.pids)


    def __getitem__(self, idx):
        """
        Return two random segments from the same trial.

        Args:
            idx (int): Index of the trial.

        Returns:
            segment (torch.Tensor): A tensor containing two segments from the same trial.
        """
        pid = self.pids[idx]
        segments = self.patient_segemtns[pid]

        # Randomly select two segments from the trial
        indices = np.random.choice(len(segments), size=2, replace=False)
        segment1 = segments[indices[0]]
        segment2 = segments[indices[1]]

        # Convert to PyTorch tensors
        segment1 = torch.tensor(segment1, dtype=torch.float32)
        segment2 = torch.tensor(segment2, dtype=torch.float32)

        segment = torch.stack((segment1, segment2), dim=0)
        
        return segment
    

class PCLR:
    def __init__(
        self,
        input_dims=12,
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
        self._proj = MLP(input_dims=output_dims, output_dims=output_dims, nlayers=2, hidden_dims=output_dims)
        
        device = torch.device(device)
        if device == torch.device("cuda") and self.multi_gpu:
            # self.net_q = nn.DataParallel(self.net_q, device_ids=gpu_idx_list)
            self._net = nn.DataParallel(self._net)
            self._proj = nn.DataParallel(self._proj)
        self._net.to(device)
        self._proj.to(device)
        # stochastic weight averaging
        # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        # self.net = self._net
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        self.proj = torch.optim.swa_utils.AveragedModel(self._proj)
        self.proj.update_parameters(self._proj)
        
        
    def fit(self, train_dataset, epochs=None, batch_size=256, lr=1e-4, wd=1.5e-6, optim="adamw", schedule=None, logdir="", checkpoint=1, verbose=1):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        params = list(self._net.parameters()) + list(self._proj.parameters())
        optimizer = self._get_optimizer(optim, params, lr, wd)
        scheduler = self._get_scheduler(schedule, optimizer, epochs, len(train_loader))
        
        epoch_loss_list = []
        start_time = datetime.now()           
        for epoch in range(epochs):
            cum_loss = 0
            for x in tqdm(train_loader, desc=f"=> Epoch {epoch+1}", leave=False):
                # count by iterations
                x = x.to(self.device)

                optimizer.zero_grad()
                
                x1, x2 = x[:, 0], x[:, 1]
                
                h1 = self._net(x1)
                z1 = self._proj(h1)
                h2 = self._net(x2)
                z2 = self._proj(h2)
                
                z1 = F.normalize(z1, dim=-1)
                z2 = F.normalize(z2, dim=-1)
                
                loss1 = self.loss_fn(z1, z2)
                loss2 = self.loss_fn(z2, z1)
                loss = (loss1 + loss2) / 2
                
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
    
    
    def _update_swa(self):
        """
        update SWA models
        """
        self.net.update_parameters(self._net)
        self.proj.update_parameters(self._proj)

        
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
    
    
    def loss_fn(self, z1, z2):
        
        
        eps = 1e-12
        sim1 = torch.mm(z1, z2.t())  # [N, N] pairs
        sim1 /= self.tau
        sim1 = torch.exp(sim1)
        
        sim2 = torch.mm(z1, z1.t())
        sim2 /= self.tau
        sim2 = torch.exp(sim2)
        sim2 = torch.triu(sim2, 1) + torch.tril(sim2, -1)
        
        denominator = torch.sum(sim1, 1) + torch.sum(sim2, 1)
        diags = torch.diag(sim1)
        
        loss = -torch.mean(torch.log((diags + eps) / (denominator + eps)))
        return loss
    
    
    def save(self, fn):
        """Save the model to a file.
        
        Args:
            fn (str): filename.
        """
        torch.save(self.net.state_dict(), fn)
    
    
    def load(self, fn):
        raise NotImplementedError("Loading a model is not implemented yet.")