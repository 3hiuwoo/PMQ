''' Train MoPa
'''
import os
import argparse
import warnings
import numpy as np
from mopa import MOPA
from data import load_data
from utils import seed_everything, get_device
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='MoPa training')
parser.add_argument('--seed', type=int, default=42, help='random seed')
# data
parser.add_argument('--root', type=str, default='dataset', help='root directory of datasets')
parser.add_argument('--data', type=str, default='chapman', help='[chapman, ptb, ptbxl]')
parser.add_argument('--length', type=int, default=300, help='length of each sample')
parser.add_argument('--overlap', type=float, default=0., help='overlap of each sample')
# model
parser.add_argument('--depth', type=int, default=10, help='number of dilated convolutional blocks')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension of the encoder')
parser.add_argument('--output_dim', type=int, default=320, help='output dimension of the encoder')
parser.add_argument('--proj_dims', type=int, default=None, help='projection head dimension, None for no projection head')
parser.add_argument('--momentum', type=float, default=0.999, help='momentum for the momentum encoder')
parser.add_argument('--queue_size', type=int, default=65536, help='queue size')
parser.add_argument('--mask_type', type=str, default='t+fb', help='opt+opt opt: [t/f/s](b/c/cb/cc)')
# training
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--shuffle', type=str, default='random', help='way to shuffle the data')
parser.add_argument('--logdir', type=str, default='log_mopa', help='directory to save weights and logs')
parser.add_argument('--checkpoint', type=int, default=1, help='frequency to save checkpoint')
parser.add_argument('--multi_gpu', action='store_true', help='whether to use multiple GPUs')
parser.add_argument('--verbose', type=int, default=1, help='if large than 0: print loss after each epoch')
# todo
# parser.add_argument('--resume', type=str, default='', help='resume training from a checkpoint')

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
    
    model = MOPA(
        input_dims=X_train.shape[-1],
        output_dims=args.output_dim,
        hidden_dims=args.hidden_dim,
        proj_dims=args.proj_dims,
        depth=args.depth,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        momentum=args.momentum,
        queue_size=args.queue_size,
        multi_gpu=args.multi_gpu
    )
    
    print(f'=> Training MoPa')
    loss_list = model.fit(
        X_train,
        y_train,
        shuffle_function=args.shuffle,
        mask_type=args.mask_type,
        epochs=args.epochs,
        logdir=logdir,
        checkpoint=args.checkpoint,
        verbose=args.verbose
        )
    # save training loss
    np.save(os.path.join(logdir, 'loss.npy'), loss_list)

if __name__ == '__main__':
    main()


    
    
    
    
    
    



