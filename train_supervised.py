'''2024-11-12

This script is used to train the model under the CMSC paradigm.

Run the script with the following command:
    python train_cmsc.py
    
See python train_cmsc.py -h for training options
'''
import os
import argparse
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric, AUROC
from tqdm import tqdm
from dataset.load import load_data
from model.load import load_model
from utils import transform
from utils.utils import set_seed, get_device, save_checkpoint

parser = argparse.ArgumentParser(description='pretraining chosen model on chosen dataset under CMSC paradigm')

parser.add_argument('--data_root', type=str, default='training2017', help='the root directory of the dataset')
parser.add_argument('--data', type=str, default='cinc2017', help='the dataset to be used')
parser.add_argument('--model', type=str, default='cnn3', help='the backbone model to be used')
parser.add_argument('--epochs', type=int, default=400, help='the number of epochs for training')
parser.add_argument('--batch_size', type=int, default=256, help='the batch size for training')
parser.add_argument('--lr', type=float, default=0.0001, help='the learning rate for training')
# parser.add_argument('--schedule', type=int, default=[100, 200, 300], help='schedule the learning rate where scale lr by 0.1')
parser.add_argument('--resume', type=str, default='', help='path to the checkpoint to be resumed')
parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
parser.add_argument('--embedding_dim', type=int, default=256, help='the dimension of the embedding in contrastive loss')
parser.add_argument('--check', type=int, default=10, help='the interval of epochs to save the checkpoint')
parser.add_argument('--log', type=str, default='log', help='the directory to save the log')
parser.add_argument('--pretrain', type=str, default='', help='path to the pretrained model')
parser.add_argument('--freeze', action='store_true', help='freeze the pretrained part of the model for linear evaluation')
parser.add_argument('--test', type=str, default='', help='path to the best model to be tested')
# parser.add_argument('--early_stop', type=int, default=20, help='stop training if the auc does not improve for n epochs')

def main():
    args = parser.parse_args()
    # directory to save the tensorboard log files and checkpoints
    prefix = 'linear_evaluation' if args.freeze else 'finetune' if args.pretrain else 'scratch'
    dir = os.path.join(args.log, f'{prefix}_{args.data}')
    
    if args.seed is not None:
        set_seed(args.seed)
        print(f'=> set seed to {args.seed}')
        
    device = get_device()
    print(f'=> using device {device}')
    
    print(f'=> creating model {args.model}')
    model = load_model(args.model, task='supervised', embeddim=args.embedding_dim)
    model.to(device)
    
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
        
    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print(f'=> loading pretrained model from {args.pretrain}')
            checkpoint = torch.load(args.pretrain, map_location=device)
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            print(f'=> no pretrained model found at {args.pretrain}')
    
    if args.freeze:
        for name, params in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                params.requires_grad = False
        
        # model.fc.weight.data.normal_(mean=0.0, std=0.01)
        # model.fc.bias.data.zero_()
        
    if args.data == 'chapman':
        trans = transform.Compose([
            # transform.Denoise(),
            transform.Normalize(),
            transform.ToTensor()
            ])
    else:
        trans = transform.ToTensor()
    
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    print(f'=> loading dataset {args.data} from {args.data_root}')
    
    train_loader, valid_loader, test_loader = load_data(root=args.data_root, dataset_name=args.data, batch_size=args.batch_size, transform=trans)
    
    print(f'=> dataset contains {len(train_loader.dataset)}|{len(valid_loader.dataset)}|{len(valid_loader.dataset)} samples')
    print(f'=> loaded with batch size of {args.batch_size}')
    
    if args.test:
        test_auc = AUROC(task='multiclass', num_classes=4).to(device)
        print(f'=> testing model from {args.test}')
        
        auc = test(test_loader, model, test_auc, device)
        print(f'=> test auc: {auc}')
        return
        
    # track loss
    train_loss = MeanMetric().to(device)
    valid_auc = AUROC(task='multiclass', num_classes=4).to(device)
    logdir = os.path.join(dir, 'log')
    writer = SummaryWriter(log_dir=logdir)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    if len(params) == 2:
        print('=> running linear evaluation for {args.epochs} epochs')
    elif args.pretrain:
        print('=> running finetune for {args.epochs} epochs')
    else:
        print('=> running train from scratch for {args.epochs} epochs')
        
    optimizer = optim.Adam(params, args.lr)
    
    best_auc = 0
    for epoch in range(start_epoch, args.epochs):
        # adjust_lr(optimizer, epoch, args.schedule)
        
        train(train_loader, model, optimizer, criterion, epoch, train_loss, writer, args.freeze, device)
        auc = validate(valid_loader, model, epoch, valid_auc, writer, device)
        
        isbest = auc > best_auc
        best_auc = max(auc, best_auc)

        if (epoch + 1) % args.check == 0 or isbest:
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            path = os.path.join(dir, 'cp', f'checkpoint_{epoch + 1}.pth')
            save_checkpoint(checkpoint, is_best=isbest, path=path)  
            
    print('=> training finished')
    writer.close()


def train(train_loader, model, optimizer, criterion, epoch, metric, writer, freeze, device):
    model.train()
    
    if freeze:
        model.eval()
    
    for signals, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
        signals = signals.to(device)
        labels = labels.to(device)
        outputs = model(signals)
        
        loss = criterion(outputs, labels)
        metric.update(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    total_loss = metric.compute()
    writer.add_scalar('loss', total_loss, epoch)
    metric.reset()
    
    
def validate(valid_loader, model, epoch, metric, writer, device):
    model.eval()
    
    with torch.no_grad():
        for signals, labels in tqdm(valid_loader, desc=f'Epoch {epoch+1}'):
            signals = signals.to(device)
            labels = labels.to(device)
            outputs = model(signals)
            
            metric.update(outputs, labels)
        
        total_metric = metric.compute()
        writer.add_scalar('auc', total_metric, epoch)
        metric.reset()
    
    return total_metric


def test(test_loader, model, metric, device):
    model.eval()
    
    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc='Testing'):
            signals = signals.to(device)
            labels = labels.to(device)
            outputs = model(signals)
            
            metric.update(outputs, labels)
        
        total_metric = metric.compute()
    
    return total_metric


if __name__ == '__main__':
    main()