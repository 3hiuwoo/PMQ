"""
See the PMQ and ETP paper for more details.
"""
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
from transformers import AutoModel, AutoTokenizer

from datautils import load_data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from encoder import TSEncoder


class ECGTextDataset(Dataset):
    def __init__(self, root="/root/autodl-tmp/dataset", name="ptbxl", length=300, overlap=0., norm=True):
        """ retrieve pairs from each patient.

        Args:
            root (str): Root directory containing the dataset.
            name (str): Dataset name (e.g., "chapman", "ptb", etc.).
            length (int): Segment length.
            overlap (float): Overlap ratio for splitting trials into segments.
            norm (bool): Whether to normalize the data.
        """
        # Load the data using the provided load_data function
        X_train, _, _, y_train, _, _, train_texts, _, _ = load_data(root=root, name=name, length=length, overlap=overlap, norm=norm)
        
        self.X_train = X_train
        self.y_train = y_train
        self.train_texts = train_texts

        self.tokenize()


    def __len__(self):
        """
        Return the number of trials.
        """
        return len(self.y_train)


    def __getitem__(self, idx):
        """
        Return two random segments from the same trial.

        Args:
            idx (int): Index of the trial.

        Returns:
            segment (torch.Tensor): A tensor containing two segments from the same trial with shape (2, length, feature).
        """
        x = self.X_train[idx]
        text = self.train_texts[idx]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(text, dtype=torch.long)
    
    
    def tokenize(self):
        """
        Tokenize the text data using the Bio_ClinicalBERT tokenizer.
        """
        join_texts = [" [SEP] ".join(text) for text in self.train_texts]
        
        tokenizer = AutoTokenizer("emilyalsentzer/Bio_ClinicalBERT")
        
        tokenized_texts = []
        for text in tqdm(join_texts, desc="=> Tokenizing ECG statements", leave=False):
            encoded = tokenizer(text, padding="max_length", max_length=128)
            tokenized_texts.append([encoded["input_ids"], encoded["attention_mask"]])
            
        self.train_texts = np.array(tokenized_texts) # [N, 2, 128]
            
        
class ETP:
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
        self._proj = nn.Linear(output_dims, output_dims)
        
        self.text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.proj = nn.Linear(768, output_dims)
            
        
        device = torch.device(device)
        if device == torch.device("cuda") and self.multi_gpu:
            # self.net_q = nn.DataParallel(self.net_q, device_ids=gpu_idx_list)
            self._net = nn.DataParallel(self._net)
            self._proj = nn.DataParallel(self._proj)
            self.text_encoder = nn.DataParallel(self.text_encoder)
            self.proj = nn.DataParallel(self.proj)
        self._net.to(device)
        self._proj.to(device)
        self.text_encoder.to(device)
        self.proj.to(device)
        # stochastic weight averaging
        # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        # self.net = self._net
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        
        
    def fit(self, train_dataset, epochs=None, batch_size=256, lr=2e-3, wd=1e-5, optim="adam", schedule=None, logdir="", checkpoint=1, verbose=1):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        params = list(self._net.parameters()) + list(self._proj.parameters())
        optimizer = self._get_optimizer(optim, params, lr, wd)
        scheduler = self._get_scheduler(schedule, optimizer, epochs, len(train_loader))
        
        epoch_loss_list = []
        start_time = datetime.now()           
        for epoch in range(epochs):
            cum_loss = 0
            for x, text in tqdm(train_loader, desc=f"=> Epoch {epoch+1}", leave=False):
                # count by iterations
                x, text = x.to(self.device), text.to(self.device)

                optimizer.zero_grad()
                
                # text features
                with torch.no_grad():
                    output = self.text_encoder(input_ids=text[:, 0], attention_mask=text[:, 1])
                    ht = output.pooler_output
                zt = self.proj(ht)
                
                # ECG features
                hx = self._net(x)
                zx = self._proj(hx)
                
                zt = F.normalize(zt, dim=-1)
                zx = F.normalize(zx, dim=-1)
                
                loss1 = self.loss_fn(zt, zx)
                loss2 = self.loss_fn(zx, zt)
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
    
    
    def loss_fn(self, q, k):
        logits = torch.mm(q, k.t())  # [N, N] pairs
        labels = torch.arange(logits.size(0), device=logits.device)  # positives are in diagonal
        loss = F.cross_entropy(logits / self.tau, labels)
        return loss
    
    
    def save(self, fn):
        """Save the model to a file.
        
        Args:
            fn (str): filename.
        """
        torch.save(self.net.state_dict(), fn)
    
    
    def load(self, fn):
        raise NotImplementedError("Loading a model is not implemented yet.")