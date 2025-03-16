import os
import argparse
import warnings
import numpy as np
from tfp import TFP2
from data import load_data
from utils import seed_everything, get_device
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='TFP training')
parser.add_argument('--seed', type=int, default=42, help='random seed')
# data
parser.add_argument('--root', type=str, default='/root/autodl-tmp/dataset', help='root directory of datasets')
parser.add_argument('--data', type=str, default='ptbxl', help='[chapman, ptb, ptbxl]')
parser.add_argument('--length', type=int, default=300, help='length of each sample')
parser.add_argument('--overlap', type=float, default=0., help='overlap of each sample')
# model
parser.add_argument('--depth', type=int, default=10, help='number of dilated convolutional blocks')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension of the encoder')
parser.add_argument('--output_dim', type=int, default=320, help='output dimension of the encoder')
parser.add_argument('--momentum', type=float, default=0.99, help='momentum for the momentum encoder')
parser.add_argument('--tau', type=float, default=0.1, help='temperature for cosine similarity')
parser.add_argument('--mask', type=str, default='bif', help='[bif, binomial, continuous, channel_binomial, channel_continuous, all_true]')
parser.add_argument('--pool', type=str, default='avg', help='[avg, max]')
# training
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--wd', type=float, default=1.5e-6, help='weight decay')
parser.add_argument('--optim', type=str, default='adamw', help='[adamw, lars]')
parser.add_argument('--schedule', type=str, default=None, help='[plateau, step, cosine, warmup, exp]')
parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--shuffle', type=str, default='random', help='way to shuffle the data')
parser.add_argument('--logdir', type=str, default='log_tfp', help='directory to save weights and logs')
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
                                       )
    
    device = get_device()
    print(f'=> Running on {device}')
    
    model = TFP2(
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
        multi_gpu=args.multi_gpu
    )
    
    print(f'=> Training TFP')
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


    
    
    
    
    
    



