'''2024-11-12

This script is used to train the model under the MCP paradigm.

Run the script with the following command:
    python train_mcp.py {options}
    
See python train_mcp.py -h for training options
'''
import os
import argparse
import torch
import numpy as np
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric
from tqdm import tqdm
from dataset.loader import load_data
from model.loader import load_model
from utils.transform import load_transforms
from utils.functional import set_seed, get_device, save_checkpoint

parser = argparse.ArgumentParser(description='pretraining chosen model on chosen dataset under MCP paradigm')

parser.add_argument('--data_root', type=str, default='trainingchapman', help='the root directory of the dataset')
parser.add_argument('--data', type=str, default='chapman', choices=['chapman', 'chapman_lead', 'chapman_trial'], help='the dataset to be used')
parser.add_argument('--model', type=str, default='res20', help='the backbone model to be used')
parser.add_argument('--epochs', type=int, default=400, help='the number of epochs for training')
parser.add_argument('--batch_size', type=int, default=256, help='the batch size for training')
parser.add_argument('--lr', type=float, default=0.0001, help='the learning rate for training')
# parser.add_argument('--schedule', type=int, default=[100, 200, 300], help='schedule the learning rate where scale lr by 0.1')
parser.add_argument('--resume', type=str, default='', help='path to the checkpoint to be resumed')
parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
parser.add_argument('--embedding_dim', type=int, default=256, help='the dimension of the embedding in contrastive loss')
parser.add_argument('--check', type=int, default=10, help='the interval of epochs to save the checkpoint')
parser.add_argument('--log', type=str, default='log', help='the directory to save the log')


def main():
    args = parser.parse_args()
    # directory to save the tensorboard log files and checkpoints
    dir = os.path.join(args.log, f'mcp_{args.model}_{args.data}_{args.batch_size}')
    # dir = args.log
    
    if args.seed is not None:
        set_seed(args.seed)
        print(f'=> set seed to {args.seed}')
        
    device = get_device()
    print(f'=> using device {device}')
    
    print(f'=> creating model {args.model}')
    model = load_model(args.model, task='mcp', embeddim=args.embedding_dim)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), args.lr)
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f'=> loading checkpoint from {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print(f'=> no checkpoint found at {args.resume}')
    else:
        start_epoch = 0
    
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        
    print(f'=> loading dataset {args.data} from {args.data_root}')
    
    # creating views
    trans = load_transforms(task='mcp', dataset_name=args.data)
    
    train_loader = load_data(root=args.data_root, task='mcp', dataset_name=args.data, batch_size=args.batch_size, transform=trans)
    
    print(f'=> dataset contains {len(train_loader.dataset)} samples')
    print(f'=> loaded with batch size of {args.batch_size}')
    
    # track loss
    loss = MeanMetric().to(device)
    
    logdir = os.path.join(dir, 'log')
    writer = SummaryWriter(log_dir=logdir)

    # queue to store patient ids
    queue_heads = np.empty(16384, dtype=str)
    ptr = 0
    
    # maybe add queue to store trial ids
    
    print(f'=> running mcp for {args.epochs} epochs')
    for epoch in range(start_epoch, args.epochs):
        # adjust_lr(optimizer, epoch, args.schedule)
        
        ptr = train(train_loader, model, optimizer, epoch, loss, writer, device, queue_heads, ptr)
        
        if (epoch + 1) % args.check == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            path = os.path.join(dir, 'cp', f'checkpoint_{epoch + 1}.pth')
            save_checkpoint(checkpoint, is_best=False, path=path)
         
    print('=> training finished')
    writer.close()


def train(train_loader, model, optimizer, epoch, metric, writer, device, queue_heads, ptr):
    model.train()
    
    bar = tqdm(train_loader, desc=f'=> Epoch {epoch+1}', leave=False)
    for signals, heads in bar:
        signals = signals.to(device)
        query_key, query_queue = model(signals)

        loss = mcp_loss(query_key, query_queue, queue_heads, heads)
            
        metric.update(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # update patient ids queue
        queue_heads[ptr:ptr+heads.shape[0]] = heads
        ptr = (ptr + heads.shape[0]) % queue_heads.shape[0]
        
        bar.set_postfix(loss=loss.item())
        
    total_loss = metric.compute()
    writer.add_scalar('loss', total_loss, epoch)
    metric.reset()
    
    return ptr


def mcp_loss(query_key, query_queue, queue_heads, heads):
    '''
    MCP loss function
    
    Args:
        query_key: product of q and k(BxB)
        query_queue: product of q and queue(BxK)
        queue_heads: patient id queue stored in the MCP model
        heads: patient ids of the current batch(B)
    '''
    # off diagonal of qk product
    pos_matrix1 = np.equal.outer(heads, heads).astype(int)
    # position of q queue product
    pos_matrix2 = np.equal.outer(heads, queue_heads).astype(int)
    
    temp = 0.1
    query_key /= temp
    query_queue /= temp
    
    query_key_exp = torch.exp(query_key)
    query_queue_exp = torch.exp(query_queue)

    denominator = torch.sum(
        torch.concat([query_key_exp, query_queue_exp], dim=1),
        dim=1)
    
    # calculate diagonal loss symmetrically
    eps = 1e-12
    diags = torch.diagonal(query_key_exp)
    loss = -torch.mean(torch.log((diags + eps)/(denominator + eps)))

    rows1, cols1 = np.where(np.triu(pos_matrix1, 1))
    if len(rows1) > 0:
        upper = query_key_exp[rows1, cols1]
        loss1 = -torch.mean(torch.log((upper + eps)/(denominator[rows1] + eps)))
        loss += loss1
 
    rows2, cols2 = np.where(pos_matrix2)
    if len(rows2) > 0:
        pos = query_queue_exp[rows2, cols2]
        loss2 = -torch.mean(torch.log((pos + eps)/(denominator[rows2] + eps)))
        loss += loss2
    
    return loss
    

if __name__ == '__main__':
    main()
