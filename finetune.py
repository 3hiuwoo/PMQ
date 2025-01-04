''' Run this script to perform partial/full finetuning or training from scratch
'''
import os
import argparse
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from model.encoder import FTClassifier
from data import load_data
from utils import seed_everything, get_device
from torchmetrics import Accuracy, F1Score, AUROC, Precision, Recall, AveragePrecision, MetricCollection


parser = argparse.ArgumentParser(description='Full/Partial Finetuning')
parser.add_argument('--seed', type=int, default=42, help='random seed')
# for the data
parser.add_argument('--root', type=str, default='dataset', help='root directory of datasets')
parser.add_argument('--data', type=str, default='chapman', help='select pretraining dataset')
parser.add_argument('--length', type=int, default=300, help='length of each sample')
# for the model
parser.add_argument('--depth', type=int, default=10, help='depth of the encoder')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension of the model')
parser.add_argument('--output_dim', type=int, default=320, help='output dimension of the model')
parser.add_argument('--p_hidden_dim', type=int, default=128, help='hidden dimension of the projection head')
parser.add_argument('--partial', action='store_true', help='partial finetuning')
parser.add_argument('--pretrain', type=str, default='', help='pretrained model weight file path')
# for the training
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--fraction', type=float, default=None, help='fraction of training data used')
parser.add_argument('--logdir', type=str, default='log', help='directory to save logs')
parser.add_argument('--checkpoint', type=int, default=1, help='save model after each checkpoint')
parser.add_argument('--multi_gpu', action='store_true', help='use multiple GPUs')
parser.add_argument('--verbose', type=int, default=1, help='print loss after each epoch')
# test
parser.add_argument('--test', type=str, default='', help='model weight file path to perform testing')
# todo
# parser.add_argument('--resume', type=str, default='', help='resume training from a checkpoint')

args = parser.parse_args()

# figure out the task
if args.pretrain:
    if args.partial:
        task = 'pft'
    else:
        task = 'fft'
else:
    task = 'scr'
        
logdir = os.path.join(args.logdir, f'{task}_{args.data}_{args.seed}')
if not os.path.exists(logdir):
    os.makedirs(logdir)
    
def main():
    seed_everything(args.seed)
    print(f'=> set seed to {args.seed}')
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(args.root, args.data, split=args.length)
    
    # only use fraction of training samples.
    if args.fraction:
        X_train = X_train[:int(X_train.shape[0] * args.fraction)]
        y_train = y_train[:int(y_train.shape[0] * args.fraction)]
        print(f'=> use {args.fraction} of training data')
    
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
    print(f'=> Running on {device}')
    
    assert X_train.ndim == 3
    device = torch.device(device)
    
    num_classes = np.unique(y_test[:, 0]).shape[0]
    model = FTClassifier(
        input_dims=X_test.shape[-1],
        output_dims=args.output_dim,
        hidden_dims=args.hidden_dim,
        p_hidden_dims=args.p_hidden_dim,
        p_output_dims=num_classes,
        depth=args.depth,
        device=device,
        multi_gpu=args.multi_gpu,
        )
    
    metrics = MetricCollection({
        'acc': Accuracy(task='multiclass', num_classes=num_classes),
        'f1': F1Score(task='multiclass', num_classes=num_classes, average='macro'), 
        'auroc': AUROC(task='multiclass', num_classes=num_classes),
        'precision': Precision(task='multiclass', num_classes=num_classes, average='macro'),
        'recall': Recall(task='multiclass', num_classes=num_classes, average='macro'),
        'auprc': AveragePrecision(task='multiclass', num_classes=num_classes) 
        }).to(device)

    if args.test:
        if os.path.isfile(args.test):
            print(f'=> test on {args.test}')
            model.load_state_dict(torch.load(args.test))
            val_metrics_dict = evaluate(model, val_loader, metrics, device)
            val_metrics_dict = {k: v.item() for k, v in val_metrics_dict.items()}
            print('metrics for validation set\n', val_metrics_dict)
            
            test_metrics_dict = evaluate(model, test_loader, metrics, device)
            test_metrics_dict = {k: v.item() for k, v in test_metrics_dict.items()}
            print('metrics for test set\n', test_metrics_dict)   
        else:
            print(f'=> find nothing in {args.test}')
        return
            
    # freeze the backbone encoder in PFT
    if args.partial:
        for name, params in model.named_parameters():
            if not name.startswith('proj_head'):
                params.requires_grad = False
       
    # todo: resume training
               
    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print(f'=> load pretrained model from {args.pretrain}')
            model.net.load_state_dict(torch.load(args.pretrain))
            
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print(f'=> number of trainable parameters groups: {len(params)}') # for debug
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    criterion = nn.CrossEntropyLoss()

    epoch_lost_list, epoch_f1_list = [], []
    
    model.train()
    for epoch in range(args.epochs):
        # training loop
        cum_loss = 0
        for x, y in tqdm(train_loader, desc=f'=> Epoch {epoch+1}', leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()

        epoch_lost_list.append(cum_loss / len(train_loader))

        # validation
        val_metrics_dict = evaluate(model, val_loader, metrics, device)
        f1 = val_metrics_dict['f1'].item()
        epoch_f1_list.append(f1)
        finetune_callback(model, epoch, f1, fraction=args.fraction, checkpoint=args.checkpoint)
        
        if args.verbose:
            print(f"=> Epoch {epoch+1} loss: {cum_loss}")
            if args.verbose > 1:
                print(val_metrics_dict)
        
    # save loss and f1score
    np.save(os.path.join(logdir, 'loss.npy'), epoch_lost_list)
    np.save(os.path.join(logdir, 'f1.npy'), epoch_f1_list)

  
def evaluate(model, loader, metrics, device):
    '''
    do validation or test
    '''
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f'=> Evaluating', leave=False):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            metrics.update(y_pred, y)
    metrics_dict = metrics.compute()
    metrics.reset()
    
    return metrics_dict


def finetune_callback(model, epoch, f1, fraction=1.0, checkpoint=1):
    if (epoch+1) == 1:
        model.finetune_f1 = f1
        torch.save(model.state_dict(), os.path.join(logdir, f'bestf1_{fraction}.pth'))
    # control the saving frequency
    if (epoch+1) % checkpoint == 0:
        if f1 > model.finetune_f1:
            model.finetune_f1 = f1
            torch.save(model.state_dict(), os.path.join(logdir, f'bestf1_{fraction}.pth'))


if __name__ == '__main__':
    main()