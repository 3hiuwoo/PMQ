import os
import argparse
import torch
import numpy as np
from torch.nn import functional as F    
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric
from tqdm import tqdm
from dataset.loader import load_data
from model.loader import load_model
from utils.functional import set_seed, get_device, save_checkpoint

parser = argparse.ArgumentParser(description='pretraining chosen model on chosen dataset under comet paradigm')

parser.add_argument('--data_root', type=str, default='trainingchapman', help='the root directory of the dataset')
parser.add_argument('--data', type=str, default='chapman', choices=['chapman'], help='the dataset to be used')
parser.add_argument('--model', type=str, default='ts', choices=['ts'], help='the backbone model to be used')
parser.add_argument('--epochs', type=int, default=100, help='the number of epochs for training')
parser.add_argument('--batch_size', type=int, default=256, help='the batch size for training')
parser.add_argument('--lr', type=float, default=0.0001, help='the learning rate for training')
# parser.add_argument('--schedule', type=int, default=[100, 200, 300], help='schedule the learning rate where scale lr by 0.1')
parser.add_argument('--resume', type=str, default='', help='path to the checkpoint to be resumed')
parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
parser.add_argument('--dim', type=int, default=256, help='the dimension of the embedding in contrastive loss')
parser.add_argument('--depth', type=int, default=10, help='the depth of the convolutional layers')
parser.add_argument('--check', type=int, default=10, help='the interval of epochs to save the checkpoint')
parser.add_argument('--log', type=str, default='log', help='the directory to save the log')


def main():
    args = parser.parse_args()
    # directory to save the tensorboard log files and checkpoints
    model_name = args.model + str(args.depth)
    dir = os.path.join(args.log, f'comet_{model_name}_{args.data}_{args.batch_size}')
    # dir = args.log
    
    if args.seed is not None:
        set_seed(args.seed)
        print(f'=> set seed to {args.seed}')
        
    device = get_device()
    print(f'=> using device {device}')
    
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        
    print(f'=> loading dataset {args.data} from {args.data_root}')
    train_loader, in_channels = load_data(root=args.data_root, task='comet', dataset_name=args.data, batch_size=args.batch_size)
    print(f'=> dataset contains {len(train_loader.dataset)} samples')
    print(f'=> loaded with batch size of {args.batch_size}')
    
    print(f'=> creating model {model_name}')
    model = load_model(task='comet', in_channels=in_channels, out_channels=args.dim, depth=args.depth)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), args.lr)
    
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
    
    # track loss
    loss = MeanMetric().to(device)
    logdir = os.path.join(dir, 'log')
    writer = SummaryWriter(log_dir=logdir)

    print(f'=> running comet for {args.epochs} epochs')
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
    
    bar = tqdm(train_loader, desc=f'=> Epoch {epoch+1}', leave=False)
    for X, y in bar:
        X = X.to(device)
        pid = y[:, 1]
        tid = y[:, 2]
        
        outputs = model(X)

        loss = comet_loss(outputs, pid, tid)

        metric.update(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        bar.set_postfix(loss=loss.item())
        
    total_loss = metric.compute()
    writer.add_scalar('loss', total_loss, epoch)
    metric.reset()
    
    
def comet_loss(outputs, pid, tid):
    '''loss function for COMET
    Args:
        outputs: the output of COMET model with shape (loss_level(normally 4), nviews(normally 2), batch_size, length, channels)
        pid: the patient ids with shape (batch_size)
        tid: the trial ids with shape (batch_size)
    '''
    factors = [0.25] * 4
    obeservation_out1 = outputs[0, 0]
    obeservation_out2 = outputs[0, 1]
    observation_loss = observation_contrastive_loss(obeservation_out1, obeservation_out2)
    
    sample_out1 = outputs[1, 0]
    sample_out2 = outputs[1, 1]
    sample_loss = sample_contrastive_loss(sample_out1, sample_out2)
    
    id_out1 = outputs[2, 0]
    id_out2 = outputs[2, 1]
    trial_loss = id_contrastive_loss(id_out1, id_out2, tid)
    patient_loss = id_contrastive_loss(id_out1, id_out2, pid)
    
    loss = factors[0] * observation_loss + factors[1] * sample_loss + factors[2] * trial_loss + factors[3] * patient_loss
    
    return loss
    

def observation_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1) # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2)) # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1] # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:] # B x 2T x (2T-1)
    logits = -F.log_softmax(logits, dim=-1) # B x 2T x (2T-1)
    
    t = torch.arange(T, device=z1.device)
    # take all samples by [:,x,x]
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


def sample_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x O x C
    z = z.transpose(0, 1)  # O x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # O x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # O x 2B x (2B-1), left-down side, remove last zero column
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]  # O x 2B x (2B-1), right-up side, remove first zero column
    logits = -F.log_softmax(logits, dim=-1)  # log softmax do dividing and log
    
    i = torch.arange(B, device=z1.device)
    # take all timestamps by [:,x,x]
    # logits[:, i, B + i - 1] : right-up, takes r_i and r_j
    # logits[:, B + i, i] : down-left, takes r_i_prime and r_j_prime
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def id_contrastive_loss(z1, z2, id):
    id = id.detach().numpy()
    pos_matrix = np.equal.outer(id, id)
    
    rows1, cols1 = np.where(np.triu(pos_matrix, 1))  # upper triangle same patient combs
    rows2, cols2 = np.where(np.tril(pos_matrix, -1))  # down triangle same patient combs

    B, T = z1.size(0), z1.size(1)
    loss = 0
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)
    # B x O x C -> B x C x O -> B x (C x O)
    view1_array = z1.permute(0, 2, 1).reshape((B, -1))
    view2_array = z2.permute(0, 2, 1).reshape((B, -1))
    norm1_vector = view1_array.norm(dim=1).unsqueeze(0)
    norm2_vector = view2_array.norm(dim=1).unsqueeze(0)
    sim_matrix = torch.mm(view1_array, view2_array.transpose(0, 1))
    norm_matrix = torch.mm(norm1_vector.transpose(0, 1), norm2_vector)
    temperature = 0.1
    argument = sim_matrix/(norm_matrix*temperature)
    sim_matrix_exp = torch.exp(argument)

    # diag_elements = torch.diag(sim_matrix_exp)

    triu_sum = torch.sum(sim_matrix_exp, 1)  # add column
    tril_sum = torch.sum(sim_matrix_exp, 0)  # add row

    """loss_diag1 = -torch.mean(torch.log(diag_elements/triu_sum))
    loss_diag2 = -torch.mean(torch.log(diag_elements/tril_sum))

    loss = loss_diag1 + loss_diag2
    loss_terms = 2"""
    loss_terms = 0

    # upper triangle same patient combs exist
    if len(rows1) > 0:
        triu_elements = sim_matrix_exp[rows1, cols1]  # row and column for upper triangle same patient combinations
        loss_triu = -torch.mean(torch.log(triu_elements / triu_sum[rows1]))
        loss += loss_triu  # technically need to add 1 more term for symmetry
        loss_terms += 1

    # down triangle same patient combs exist
    if len(rows2) > 0:
        tril_elements = sim_matrix_exp[rows2, cols2]  # row and column for down triangle same patient combinations
        loss_tril = -torch.mean(torch.log(tril_elements / tril_sum[cols2]))
        loss += loss_tril  # technically need to add 1 more term for symmetry
        loss_terms += 1

    if loss_terms == 0:
        return 0
    else:
        loss = loss/loss_terms
        return loss


if __name__ == '__main__':
    main()