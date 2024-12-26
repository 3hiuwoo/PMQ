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

parser = argparse.ArgumentParser(description='pretraining chosen model on chosen dataset under comet paradigm')

parser.add_argument('--data_root', type=str, default='trainingchapman', help='the root directory of the dataset')
parser.add_argument('--data', type=str, default='chapman_trial', choices=['chapman_trial'], help='the dataset to be used')
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
    dir = os.path.join(args.log, f'comet_{args.model}_{args.data}_{args.batch_size}')
    # dir = args.log
    
    if args.seed is not None:
        set_seed(args.seed)
        print(f'=> set seed to {args.seed}')
        
    device = get_device()
    print(f'=> using device {device}')
    
    print(f'=> creating model {args.model}')
    if args.data == 'chapman_trial':
        in_channels = 12
    model = load_model(args.model, task='comet', in_channels=in_channels, embeddim=args.embedding_dim)
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
    
    # creating views for different levels
    trans = load_transforms(task='comet', dataset_name=args.data)
    
    train_loader = load_data(root=args.data_root, task='comet', dataset_name=args.data, batch_size=args.batch_size, transform=trans)
    
    print(f'=> dataset contains {len(train_loader.dataset)} samples')
    print(f'=> loaded with batch size of {args.batch_size}')
    
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
    for signals, heads, trials in bar:
        signals = signals.to(device)
        outputs = model(signals)

        loss = comet_loss(outputs, heads, trials)

        metric.update(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        bar.set_postfix(loss=loss.item())
        
    total_loss = metric.compute()
    writer.add_scalar('loss', total_loss, epoch)
    metric.reset()
    
    
def comet_loss(outputs, heads, trials):
    '''loss function for COMET
    Args:
        outputs: the output of COMET model with shape (loss_level(normally 4), nviews(normally 2), batch_size, channels, length)
        heads: the patient ids with shape (batch_size)
        trials: the trial ids with shape (batch_size)
    '''
    B = outputs.shape[2] # batch size
    T = outputs.shape[-1] # time length
    
    # ----------observation loss----------
    oout = outputs[0]
    z1 = torch.cat([oout[0], oout[1]], dim=2)  # B x C x 2T
    osim = torch.matmul(z1.transpose(1, 2), z1)  # B x 2T x 2T
    ologits = torch.tril(osim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    ologits += torch.triu(osim, diagonal=1)[:, :, 1:]
    ologits = -torch.nn.functional.log_softmax(ologits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    # take all samples by [:,x,x]
    oloss = (ologits[:, t, T + t - 1].mean() + ologits[:, T + t, t].mean()) / 2
    
    
    # ----------sample loss----------
    zout = outputs[1]
    z2 = torch.cat([zout[0], zout[1]], dim=0)  # 2B x C x T
    z2 = z2.permute(2, 0, 1)  # T x 2B x C
    ssim = torch.matmul(z2, z2.transpose(1, 2))  # O x 2B x 2B
    slogits = torch.tril(ssim, diagonal=-1)[:, :, :-1]    # O x 2B x (2B-1), left-down side, remove last zero column
    slogits += torch.triu(ssim, diagonal=1)[:, :, 1:]  # O x 2B x (2B-1), right-up side, remove first zero column
    slogits = -torch.nn.functional.log_softmax(slogits, dim=-1)  # log softmax do dividing and log
    
    i = torch.arange(B, device=z2.device)
    # take all timestamps by [:,x,x]
    # logits[:, i, B + i - 1] : right-up, takes r_i and r_j
    # logits[:, B + i, i] : down-left, takes r_i_prime and r_j_prime
    zloss = (slogits[:, i, B + i - 1].mean() + slogits[:, B + i, i].mean()) / 2
    
    # ----------trial loss----------
    tout = outputs[2]
    trials = np.array(trials)
    tpos = np.equal.outer(trials, trials).astype(int)
    
    # get normalized embeddings for each view
    t1 = tout.reshape(B, -1)
    t1 = torch.nn.functional.normalize(t1, dim=-1)
    t2 = tout.reshape(B, -1)
    t2 = torch.nn.functional.normalize(t2, dim=-1)
    
    # calculate the similarity matrix
    tsim = torch.matmul(t1, t2.T)
    tsim /= 0.1
    tsim_exp = torch.exp(tsim)

    # sum over similarities across rows and columns
    trow_sum = torch.sum(tsim_exp, dim=1)
    tcol_sum = torch.sum(tsim_exp, dim=0)
    
    # calculate diagonal loss symmetrically
    eps = 1e-12
    # diags = torch.diagonal(sim_matrix_exp)
    # lossd1 = -torch.mean(torch.log((diags + eps)/(row_sum + eps)))
    # lossd2 = -torch.mean(torch.log((diags + eps)/(col_sum + eps)))
    # loss = lossd1 + lossd2
   
    tloss = 0
    tloss_term = 0
    # calculate off-diagonal loss symmetrically
    tupper_rows, tupper_cols = np.where(np.triu(tpos, 1))
    tlower_rows, tlower_cols = np.where(np.tril(tpos, -1))
    
    if len(tupper_rows) > 0:
        tupper = tsim_exp[tupper_rows, tupper_cols]
        tlossou = -torch.mean(torch.log((tupper + eps)/(trow_sum[tupper_rows] + eps)))
        tloss += tlossou
        tloss_term += 1

    if len(tlower_cols) > 0:
        tlower = tsim_exp[tlower_rows, tlower_cols]
        tlossol = -torch.mean(torch.log((tlower + eps)/(tcol_sum[tlower_cols] + eps)))
        tloss += tlossol
        tloss_term += 1
    
    tloss /= tloss_term
    
    
    # ----------patient loss----------
    pout = outputs[3]
    heads = np.array(heads)
    ppos = np.equal.outer(heads, heads).astype(int)
    
    # get normalized embeddings for each view
    p1 = pout[0].reshape(B, -1)
    p1 = torch.nn.functional.normalize(p1, dim=-1)
    p2 = pout[1].reshape(B, -1)
    p2 = torch.nn.functional.normalize(p2, dim=-1)
    
    # calculate the similarity matrix
    psim = torch.matmul(p1, p2.T)
    psim /= 0.1
    psim_exp = torch.exp(psim)

    # sum over similarities across rows and columns
    prow_sum = torch.sum(psim_exp, dim=1)
    pcol_sum = torch.sum(psim_exp, dim=0)
    
    # calculate diagonal loss symmetrically
    # diags = torch.diagonal(sim_matrix_exp)
    # lossd1 = -torch.mean(torch.log((diags + eps)/(row_sum + eps)))
    # lossd2 = -torch.mean(torch.log((diags + eps)/(col_sum + eps)))
    # loss = lossd1 + lossd2
   
    ploss = 0
    ploss_term = 0
    # calculate off-diagonal loss symmetrically
    pupper_rows, pupper_cols = np.where(np.triu(ppos, 1))
    plower_rows, plower_cols = np.where(np.tril(ppos, -1))
    
    if len(pupper_rows) > 0:
        pupper = psim_exp[pupper_rows, pupper_cols]
        plossou = -torch.mean(torch.log((pupper + eps)/(prow_sum[pupper_rows] + eps)))
        ploss += plossou
        ploss_term += 1

    if len(plower_cols) > 0:
        plower = psim_exp[plower_rows, plower_cols]
        plossol = -torch.mean(torch.log((plower + eps)/(pcol_sum[plower_cols] + eps)))
        ploss += plossol
        ploss_term += 1
    
    ploss /= ploss_term
    
    
    # ----------total loss----------
    factors = [0.25] * 4
    loss = oloss * factors[0] + zloss * factors[1] + tloss * factors[2] + ploss * factors[3]
    
    return loss


if __name__ == '__main__':
    main()