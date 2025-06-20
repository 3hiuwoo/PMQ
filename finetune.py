import argparse
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import (AUROC, Accuracy, AveragePrecision, F1Score,
                          MetricCollection, Precision, Recall)
from tqdm import tqdm

from data import load_data
from encoder import FTClassifier
from utils import get_device, seed_everything, start_logging, stop_logging

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Fine-tuning/Training from scratch")
parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45], help="list of random seeds")
# data
parser.add_argument("--root", type=str, default="/root/autodl-tmp/dataset", help="root directory of datasets")
parser.add_argument("--datas", type=str, nargs="+", default=["ptbxl", "chapman", "cpsc2018"], help="downstream dataset: [ptb, ptbxl, chapman, cpsc2018]")
parser.add_argument("--length", type=int, default=300, help="length of each sample")
parser.add_argument("--overlap", type=float, default=0., help="overlap of each sample")
# model
parser.add_argument("--depth", type=int, default=10, help="number of dilated convolutional blocks")
parser.add_argument("--hidden_dim", type=int, default=64, help="output dimension of input projector")
parser.add_argument("--output_dim", type=int, default=320, help="output dimension of the encoder")
parser.add_argument("--p_hidden_dim", type=int, default=128, help="hidden dimension of the projection head")
parser.add_argument("--pretrain", type=str, default="", help="encoder weight file path, if None, train from scratch")
parser.add_argument("--pool", type=str, default="avg", help="pooling method: [avg, max]")
# training
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--fractions", type=float, nargs="+", default=[0.3, 0.1, 0.01], help="list of fractions of training data")
parser.add_argument("--freeze", action="store_true", help="whether to partial fine-tuning")
parser.add_argument("--logdir", type=str, default="log", help="directory to save logs")
parser.add_argument("--multi_gpu", action="store_true", help="whether to use multiple GPUs")
parser.add_argument("--verbose", type=int, default=1, help="0: no print, 1: print loss, 2: print test metrics, 3: print all metrics")
parser.add_argument("--ensemble", action="store_true", help="whether to ensemble the predictions")

args = parser.parse_args()

def main():
    # figure out the task
    if args.pretrain:
        if args.freeze:
            task = "pft"
        else:
            task = "fft"
    else:
        task = "scratch"
        
    print("=> Arguments:", vars(args))
            
    logdir = os.path.join(args.logdir, f"{task}")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    print(f"=> Weights and logs will be saved in {logdir}")

    # save argumens information
    with open(os.path.join(logdir, "args.txt"), "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    
    print(f"=> Running cross {len(args.datas)} dataset, {len(args.seeds)} seeds and {len(args.fractions)} fractions")
    for data in args.datas:
        subdir = os.path.join(logdir, f"{data}")
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        X_train, X_val, X_test,\
        y_train, y_val, y_test = load_data(root=args.root,
                                           name=data,
                                           length=args.length,
                                           overlap=args.overlap,
                                           ensemble=args.ensemble,
                                           shuffle_seed=args.seeds[1])
        print(f"=> Data shape: {X_train.shape}, {X_val.shape}, {X_test.shape}")
        for seed in args.seeds:
            seed_everything(seed)
            print(f"=> Set seed to {seed}")
            for fraction in args.fractions:
                run(subdir, fraction, seed, X_train, X_val, X_test, y_train, y_val, y_test)

        print(f"==================== Calculating total metrics ====================")
        start_logging("total", subdir) # simultaneously save the print out to file
        for fraction in args.fractions:
            print(f"=> Fraction: {fraction}, Seeds: {args.seeds}")
            val_path = os.path.join(subdir, f"val_{fraction}.csv")
            test_path = os.path.join(subdir, f"test_{fraction}.csv")
            val_df = pd.read_csv(val_path, index_col=0)
            test_df = pd.read_csv(test_path, index_col=0)
            val_mean = val_df.mean().to_dict()
            test_mean = test_df.mean().to_dict()
            val_std = val_df.std().to_dict()
            test_std = test_df.std().to_dict()
            
            val_out = (f"{k}: {m:.6f}±{s:.6f}" for k, m, s in zip(val_mean.keys(), val_mean.values(), val_std.values()))
            test_out = (f"{k}: {m:.6f}±{s:.6f}" for k, m, s in zip(test_mean.keys(), test_mean.values(), test_std.values()))
            print("=> Metrics for validation set\n", "\n".join(val_out), "\n")
            print("=> Metrics for test set\n", "\n".join(test_out), "\n")
        stop_logging()


def run(logdir, fraction, seed, X_train, X_val, X_test, y_train, y_val, y_test):
    """ Run for one random seed and one fraction of training data
    
    Args:
        logdir (str): directory to save logs
        fraction (float): fraction of training data
        seed (int): random seed
        X_train (np.ndarray): training data
        X_val (np.ndarray): validation data
        X_test (np.ndarray): test data
        y_train (np.ndarray): training labels
        y_val (np.ndarray): validation labels
        y_test (np.ndarray): test labels
    """
    
    # only use fraction of training samples.
    if fraction < 1:
        X_train = X_train[:int(X_train.shape[0] * fraction)]
        y_train = y_train[:int(y_train.shape[0] * fraction)]
        print(f"=> Using {fraction*100}% of training data")
    
    train_dataset = TensorDataset(torch.from_numpy(X_train).to(torch.float),
                                  torch.from_numpy(y_train[:, 0]).to(torch.long))
    val_dataset = TensorDataset(torch.from_numpy(X_val).to(torch.float),
                                torch.from_numpy(y_val[:, 0]).to(torch.long))
    test_dataset = TensorDataset(torch.from_numpy(X_test).to(torch.float),
                                 torch.from_numpy(y_test[:, 0]).to(torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    device = get_device()
    print(f"=> Running on {device}")
    device = torch.device(device)
    
    input_dims = X_test.shape[-1]
    num_classes = np.unique(y_test[:, 0]).shape[0]
    model = FTClassifier(
        input_dims=input_dims,
        output_dims=args.output_dim,
        hidden_dims=args.hidden_dim,
        depth=args.depth,
        p_hidden_dims=args.p_hidden_dim,
        p_output_dims=num_classes,
        pool=args.pool,
        device=device,
        multi_gpu=args.multi_gpu,
        )
    
    metrics = MetricCollection({
        "acc": Accuracy(task="multiclass", num_classes=num_classes),
        "f1": F1Score(task="multiclass", num_classes=num_classes, average="macro"), 
        "auroc": AUROC(task="multiclass", num_classes=num_classes),
        "precision": Precision(task="multiclass", num_classes=num_classes, average="macro"),
        "recall": Recall(task="multiclass", num_classes=num_classes, average="macro"),
        "auprc": AveragePrecision(task="multiclass", num_classes=num_classes) 
        }).to(device)
           
    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print(f"=> Loading pretrained model from {args.pretrain}")
            weights = torch.load(args.pretrain)
            msg = model.net.load_state_dict(weights, strict=False)
            print("=>", msg)
        else:
            print(f"=> Find nothing in {args.pretrain}")
            
        if args.freeze:
            model.proj_head = nn.Linear(args.output_dim, num_classes).to(device)
            for name, param in model.net.named_parameters():
                param.requires_grad = False
            for name, param in model._net.named_parameters():
                param.requires_grad = False
            print(f"=> Freeze the encoder and append a linear layer")
    else:
        print(f"=> Training from scratch")
            
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if args.pretrain and args.freeze:
        assert len(params) == 2 # weight, bias
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()

    epoch_lost_list, epoch_f1_list = [], []
    start_time = datetime.now()
 
    for epoch in range(args.epochs):
        # training loop
        loss = train(model, train_loader, optimizer, criterion, epoch, device)
        epoch_lost_list.append(loss)
        
        # validation
        val_metrics_dict = evaluate(model, val_loader, metrics, device, ensemble=args.ensemble)
        f1 = val_metrics_dict["f1"]
        epoch_f1_list.append(f1)
        
        # save the model with best F1
        finetune_callback(logdir, model, epoch, f1, fraction, seed)
        
        if args.verbose:
            print(f"=> Epoch {epoch+1} loss: {loss}")
            if args.verbose > 2:
                print(val_metrics_dict)
                
    end_time = datetime.now()
    print(f"=> Training finished in {end_time - start_time}")
    
    np.save(os.path.join(logdir, f"loss_{fraction}_{seed}.npy"), epoch_lost_list)
    np.save(os.path.join(logdir, f"f1_{fraction}_{seed}.npy"), epoch_f1_list)
    
    # testing
    test_path = os.path.join(logdir, f"bestf1_{fraction}_{seed}.pth")
    
    start_logging(seed, logdir) # simultaneously save the print out to file
    print(f"=> Testing on {test_path}")
    model.load_state_dict(torch.load(test_path))
    
    val_metrics_dict = evaluate(model, val_loader, metrics, device, ensemble=args.ensemble)
    if args.verbose > 1:
        print("=> Metrics for validation set\n", val_metrics_dict)
    
    test_metrics_dict = evaluate(model, test_loader, metrics, device, ensemble=args.ensemble)
    if args.verbose > 1:
        print("=> Metrics for test set\n", test_metrics_dict)
    stop_logging(logdir, seed, fraction, val_metrics_dict, test_metrics_dict)
    
    del model, optimizer, criterion, metrics, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()


def train(model, loader, optimizer, criterion, epoch, device):
    """ one epoch training. """
    model.train()
    
    cum_loss = 0
    for x, y in tqdm(loader, desc=f"=> Epoch {epoch+1}", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        cum_loss += loss.item()

    cum_loss /= len(loader)
    return cum_loss
    
    
def evaluate(model, loader, metrics, device, ensemble=False):
    """ do validation or test. """
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"=> Evaluating", leave=False):
            x, y = x.to(device), y.to(device)
            if ensemble:
                # ensemble the predictions
                B, S, T, F = x.shape
                x = x.view(-1, T, F)
                logits = model(x)
                y_pred = logits.view(B, S, -1).mean(dim=1)
            else:
                # no ensemble
                y_pred = model(x)
            metrics.update(y_pred, y)
    metrics_dict = metrics.compute()
    metrics.reset()
    
    metrics_dict = {k: v.item() for k, v in metrics_dict.items()}
    return metrics_dict


def finetune_callback(logdir, model, epoch, f1, fraction, seed):
    """
    save the model with best F1
    """
    if (epoch+1) == 1:
        model.finetune_f1 = f1
        torch.save(model.state_dict(), os.path.join(logdir, f"bestf1_{fraction}_{seed}.pth"))
    if f1 > model.finetune_f1:
        model.finetune_f1 = f1
        torch.save(model.state_dict(), os.path.join(logdir, f"bestf1_{fraction}_{seed}.pth"))

if __name__ == "__main__":
    main()