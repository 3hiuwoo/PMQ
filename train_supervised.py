'''2024-11-13

This script is used to train the model under the supervised paradigm.
Run the script for finetuning/linear evaluation with pretrained model or training from scratch.

For training from scratch, run the script with the following command:
    python train_supervised.py {options}
    
For finetuning/linear evaluation with pretrained model, run the script with the following command:
    python train_supervised.py --pretrain {path_to_pretrained_model} (--freeze) {options}
    
See python train_supervised.py -h for training options
'''
import os
import argparse
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric, AUROC, Accuracy, F1Score, MetricCollection
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
    prefix = 'lineval' if args.freeze else 'finetune' if args.pretrain else 'scratch'
    dirname = f'{prefix}_{args.model}_{args.data}_{args.batch_size}'
    
    if args.pretrain:
        postfix = args.pretrain.split(os.sep)[-3]
        dirname += f'__{postfix}'
        filename = args.pretrain.split(os.sep)[-1].split('.')[0]
        num_epochs = int(filename.split('_')[-1])
        dirname += f'_{num_epochs}'

    dir = os.path.join(args.log, dirname)
    # dir = args.log
        
    if args.seed is not None:
        set_seed(args.seed)
        print(f'=> set seed to {args.seed}')
        
    device = get_device()
    print(f'=> using device {device}')
    
    print(f'=> creating model with {args.model}')
    model = load_model(args.model, task='supervised', embeddim=args.embedding_dim)
    model.to(device)
    
    if args.freeze:
        for name, params in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                params.requires_grad = False
        
        # model.fc.weight.data.normal_(mean=0.0, std=0.01)
        # model.fc.bias.data.zero_()
        
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(params, args.lr)
    
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
        
    if args.data == 'chapman':
        trans = transform.Compose([
            # transform.Denoise(),
            transform.DownSample(2),
            transform.Normalize(),
            transform.ToTensor()
            ])
    else:
        trans = transform.ToTensor()
    
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    print(f'=> loading dataset {args.data} from {args.data_root}')
    
    train_loader, valid_loader, test_loader = load_data(root=args.data_root, task='supervised', dataset_name=args.data, batch_size=args.batch_size, transform=trans)
    
    print(f'=> dataset contains {len(train_loader.dataset)}|{len(valid_loader.dataset)}|{len(valid_loader.dataset)} samples')
    print(f'=> loaded with batch size of {args.batch_size}')
    
    if args.test:
        if os.path.isfile(args.test):
            print(f'=> testing model from {args.test}')
            checkpoint = torch.load(args.test, map_location=device)
            model.load_state_dict(checkpoint['model'])
            
            test_metrics = MetricCollection({
                'acc': Accuracy(task='multiclass', num_classes=4),
                'auc': AUROC(task='multiclass', num_classes=4),
                'f1': F1Score(task='multiclass', num_classes=4, average='macro')}).to(device)
            metrics = test(test_loader, model, test_metrics, device)
            print(f'=> auc: {metrics["auc"].item()}, acc: {metrics["acc"].item()}, f1: {metrics["f1"].item()}')
            return
        
        else:
            print(f'=> no model found at {args.test}')
            return
    
    # track loss
    train_loss = MeanMetric().to(device)
    valid_metrics = MetricCollection({
        'acc': Accuracy(task='multiclass', num_classes=4), 
        'auc': AUROC(task='multiclass', num_classes=4),
        'f1': F1Score(task='multiclass', num_classes=4, average='macro')}).to(device)
    
    logdir = os.path.join(dir, 'log')
    writer = SummaryWriter(log_dir=logdir)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)

    if len(params) == 2:
        print(f'=> running linear evaluation for {args.epochs} epochs')
    elif args.pretrain:
        print(f'=> running finetune for {args.epochs} epochs')
    else:
        print(f'=> running train from scratch for {args.epochs} epochs')

    best_auc = 0
    for epoch in range(start_epoch, args.epochs):
        # adjust_lr(optimizer, epoch, args.schedule)
        
        train(train_loader, model, optimizer, criterion, epoch, train_loss, writer, args.freeze, device)
        metrics = validate(valid_loader, model, epoch, valid_metrics, writer, device)
        
        auc = metrics['auc']
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
    
    for signals, labels in tqdm(train_loader, desc=f'=> Epoch {epoch+1}', leave=False):
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
    
    
def validate(valid_loader, model, epoch, metrics, writer, device):
    model.eval()
    
    with torch.no_grad():
        for signals, labels in tqdm(valid_loader, desc=f'=> Validating', leave=False):
            signals = signals.to(device)
            labels = labels.to(device)
            outputs = model(signals)
            
            metrics.update(outputs, labels)
        
        total_metrics = metrics.compute()
        writer.add_scalars('metrics', total_metrics, epoch)
        metrics.reset()
    
    return total_metrics


def test(test_loader, model, metrics, device):
    model.eval()
    
    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc='=> Testing'):
            signals = signals.to(device)
            labels = labels.to(device)
            outputs = model(signals)
            
            metrics.update(outputs, labels)
        
        total_metrics = metrics.compute()
    
    return total_metrics


if __name__ == '__main__':
    main()