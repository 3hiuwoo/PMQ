''' 
TODO1: moco without patient loss and use random shuffle -> omit --patient
TODO2: time mask probability 0, 0.25, 0.5(now), 0.75, 1(NA) -> new branch
TODO3: freq mask probability 0, 0.1(now), 0.2, 0.5, ... -> pass mask=binomial for 0, new branch for others
TODO4: no temporal neighbor -> omit --neighbor and pass --length=300

'''
import os
import argparse
import warnings
import numpy as np
from model import MoCoP, MoCoQ
from data import load_data
from utils import seed_everything, get_device
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='TFP training')
parser.add_argument('--seed', type=int, default=42, help='random seed')
# data
parser.add_argument('--root', type=str, default='/root/autodl-tmp/dataset', help='root directory of datasets')
parser.add_argument('--data', type=str, default='ptbxl', help='[chapman, ptb, ptbxl]')
parser.add_argument('--length', type=int, default=600, help='length of each sample')
parser.add_argument('--overlap', type=float, default=0., help='overlap of each sample')
parser.add_argument('--neighbor', action='store_true', help='whether to use neighbor samples')
# model
parser.add_argument('--depth', type=int, default=10, help='number of dilated convolutional blocks')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension of the encoder')
parser.add_argument('--output_dim', type=int, default=320, help='output dimension of the encoder')
parser.add_argument('--momentum', type=float, default=0.999, help='momentum for the momentum encoder')
parser.add_argument('--tau', type=float, default=0.1, help='temperature for cosine similarity')
parser.add_argument('--mask', type=str, default='binomialf', help='{binomial, continuous, channel_binomial, channel_continuous, all_true}(f)')
parser.add_argument('--pool', type=str, default='avg', help='[avg, max]')
parser.add_argument('--queue_size', type=int, default=16384, help='queue size for the momentum encoder')
parser.add_argument('--patient', action='store_true', help='whether to use patient loss')
# training
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--wd', type=float, default=1.5e-6, help='weight decay')
parser.add_argument('--optim', type=str, default='adamw', help='[adamw, lars]')
parser.add_argument('--schedule', type=str, default=None, help='[plateau, step, cosine, warmup, exp]')
parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--shuffle', type=str, default='trial', help='way to shuffle the data')
parser.add_argument('--logdir', type=str, default='log', help='directory to save weights and logs')
parser.add_argument('--checkpoint', type=int, default=1, help='frequency to save checkpoint')
parser.add_argument('--multi_gpu', action='store_true', help='whether to use multiple GPUs')
parser.add_argument('--verbose', type=int, default=1, help='if large than 0: print loss after each epoch')

def main(): 
    args = parser.parse_args()
    print('=> Arguments:', vars(args))
    
    logdir = os.path.join(args.logdir, f'pretrain_{args.data}_{args.seed}')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
        
    print(f'=> Weights will be saved in {logdir}')
    
    with open(os.path.join(logdir, 'args.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')
        
    seed_everything(args.seed)
    print(f'=> Set seed to {args.seed}')
    
    X_train, X_val, X_test,\
    y_train, y_val, y_test = load_data(args.root,
                                       args.data,
                                       length=args.length,
                                       overlap=args.overlap,
                                       neighbor=args.neighbor
                                       )
    
    device = get_device()
    print(f'=> Running on {device}')
    
    if args.queue_size:
        print(f'=> Using MoCo-Q')
        model = MoCoQ(
            input_dims=X_train.shape[-1],
            output_dims=args.output_dim,
            hidden_dims=args.hidden_dim,
            depth=args.depth,
            pool=args.pool,
            device=device,
            lr=args.lr,
            batch_size=args.batch_size,
            momentum=args.momentum,
            tau=args.tau,
            wd=args.wd,
            queue_size=args.queue_size,
            multi_gpu=args.multi_gpu,
            use_patient=args.patient
        )
    else:
        print(f'=> Using MoCo-P')
        model = MoCoP(
            input_dims=X_train.shape[-1],
            output_dims=args.output_dim,
            hidden_dims=args.hidden_dim,
            depth=args.depth,
            pool=args.pool,
            device=device,
            lr=args.lr,
            batch_size=args.batch_size,
            momentum=args.momentum,
            tau=args.tau,
            wd=args.wd,
            multi_gpu=args.multi_gpu,
            use_patient=args.patient
        )
        
    loss_list = model.fit(
        X_train,
        y_train,
        shuffle_function=args.shuffle,
        mask=args.mask,
        epochs=args.epochs,
        optim=args.optim,
        schedule=args.schedule,
        logdir=logdir,
        checkpoint=args.checkpoint,
        verbose=args.verbose
        )
    # save training loss
    np.save(os.path.join(logdir, 'loss.npy'), loss_list)

if __name__ == '__main__':
    main()


    
    
    
    
    
    



