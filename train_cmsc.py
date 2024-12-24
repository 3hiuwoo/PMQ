'''2024-11-12

This script is used to train the model under the CMSC paradigm.

Run the script with the following command:
    python train_cmsc.py {options}
    
See python train_cmsc.py -h for training options
'''
import os
import argparse
import torch
import numpy as np
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric
from tqdm import tqdm
from dataset.load import load_data
from model.load import load_model
from utils import transform
from utils.utils import set_seed, get_device, save_checkpoint

parser = argparse.ArgumentParser(description='pretraining chosen model on chosen dataset under CMSC paradigm')

parser.add_argument('--data_root', type=str, default='trainingchapman', help='the root directory of the dataset')
parser.add_argument('--data', type=str, default='chapman', help='the dataset to be used')
parser.add_argument('--model', type=str, default='res20', help='the backbone model to be used')
parser.add_argument('--epochs', type=int, default=400, help='the number of epochs for training')
parser.add_argument('--batch_size', type=int, default=256, help='the batch size for training')
parser.add_argument('--lr', type=float, default=0.0001, help='the learning rate for training')
# parser.add_argument('--schedule', type=int, default=[100, 200, 300], help='schedule the learning rate where scale lr by 0.1')
parser.add_argument('--resume', type=str, default='', help='path to the checkpoint to be resumed')
parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
# parser.add_argument('--embedding_dim', type=int, default=256, help='the dimension of the embedding in contrastive loss')
parser.add_argument('--check', type=int, default=10, help='the interval of epochs to save the checkpoint')
parser.add_argument('--log', type=str, default='log', help='the directory to save the log')


def main():
    args = parser.parse_args()
    # directory to save the tensorboard log files and checkpoints
    dir = os.path.join(args.log, f'cmsc_{args.model}_{args.data}_{args.batch_size}')
    # dir = args.log
    
    if args.seed is not None:
        set_seed(args.seed)
        print(f'=> set seed to {args.seed}')
        
    device = get_device()
    print(f'=> using device {device}')
    
    print(f'=> creating model {args.model}')
    model = load_model(args.model, task='cmsc', embeddim=args.embedding_dim)
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
    
    # creating views
    trans = transform.Compose([
        # transform.Denoise(),
        transform.Normalize(),
        transform.CreateView(transform.Segment()),
        transform.ToTensor()
        ])
    
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        
    print(f'=> loading dataset {args.data} from {args.data_root}')
    
    train_loader = load_data(root=args.data_root, task='cmsc', dataset_name=args.data, batch_size=args.batch_size, transform=trans)
    
    print(f'=> dataset contains {len(train_loader.dataset)} samples')
    print(f'=> loaded with batch size of {args.batch_size}')
    
    # track loss
    loss = MeanMetric().to(device)
    
    logdir = os.path.join(dir, 'log')
    writer = SummaryWriter(log_dir=logdir)

    print(f'=> running cmsc for {args.epochs} epochs')
    for epoch in range(start_epoch, args.epochs):
        # adjust_lr(optimizer, epoch, args.schedule)
        
        train(train_loader, model, optimizer, epoch, loss, writer, device)
        
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


def train(train_loader, model, optimizer, epoch, metric, writer, device):
    model.train()
    
    for signals, heads in tqdm(train_loader, desc=f'=> Epoch {epoch+1}', leave=False):
        signals = signals.to(device)
        outputs = model(signals)

        loss = cmsc_loss(outputs, heads)

        metric.update(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    total_loss = metric.compute()
    writer.add_scalar('loss', total_loss, epoch)
    metric.reset()
    
    
def cmsc_loss(outputs, heads):
    '''
    fix number of views to 2
    
    Args:
        outputs: embedding for each view (NxBxH)
        heads: the head of sample in the batch (B)
        
    Returns:
        the contrastive loss cross patients including 4 terms:
        2 symmetric diagonal terms and 2 symmetric off-diagonal terms
    '''
    # find the diagonal and off-diagonal positions that need to calculate the loss
    heads = np.array(heads)
    pos_matrix = np.equal.outer(heads, heads).astype(int)
    
    # get normalized embeddings for each view
    view1 = outputs[0]
    view1 = torch.nn.functional.normalize(view1, dim=-1)
    view2 = outputs[1]
    view2 = torch.nn.functional.normalize(view2, dim=-1)
    
    # calculate the similarity matrix
    sim_matrix = torch.matmul(view1, view2.T)
    sim_matrix /= 0.1
    sim_matrix_exp = torch.exp(sim_matrix)

    # sum over similarities across rows and columns
    row_sum = torch.sum(sim_matrix_exp, dim=1)
    col_sum = torch.sum(sim_matrix_exp, dim=0)
    
    # calculate diagonal loss symmetrically
    eps = 1e-12
    diags = torch.diagonal(sim_matrix_exp)
    lossd1 = -torch.mean(torch.log((diags + eps)/(row_sum + eps)))
    lossd2 = -torch.mean(torch.log((diags + eps)/(col_sum + eps)))
    loss = lossd1 + lossd2
   
    # calculate off-diagonal loss symmetrically
    upper_rows, upper_cols = np.where(np.triu(pos_matrix, 1))
    lower_rows, lower_cols = np.where(np.tril(pos_matrix, -1))
    if len(upper_rows) > 0:
        upper = sim_matrix_exp[upper_rows, upper_cols]
        lossou = -torch.mean(torch.log((upper + eps)/(row_sum[upper_rows] + eps)))
        loss += lossou

    if len(lower_cols) > 0:
        lower = sim_matrix_exp[lower_rows, lower_cols]
        lossol = -torch.mean(torch.log((lower + eps)/(col_sum[lower_cols] + eps)))
        loss += lossol
    
    return loss


if __name__ == '__main__':
    main()