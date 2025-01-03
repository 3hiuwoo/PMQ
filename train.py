import os
import argparse
import torch
import numpy as np
from mcp import MCP
from data import load_data
from utils import seed_everything, get_device


parser = argparse.ArgumentParser(description='MCP training')
parser.add_argument('--seed', type=int, default=42, help='random seed')
# for the data
parser.add_argument('--root', type=str, default='data', help='root directory of datasets')
parser.add_argument('--data', type=str, default='chapman', help='select pretraining dataset')
parser.add_argument('--length', type=int, default=300, help='length of each sample')
# for the model
parser.add_argument('--depth', type=int, default=10, help='depth of the encoder')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension of the model')
parser.add_argument('--output_dim', type=int, help='output dimension of the model')
parser.add_argument('--momentum', type=float, default=0.999, help='momentum for the model')
parser.add_argument('--queue_size', type=int, default=16384, help='queue size for the model')
parser.add_argument('--num_queues', type=int, default=1, help='number of queues for the model')
parser.add_argument('--masks', type=str, default=['all_true', 'all_true', 'continuous', 'continuous'], nargs='*', help='masks for the model')
parser.add_argument('--factors', type=int, default=[0.25, 0.25, 0.25, 0.25], nargs='*', help='factors for each level')
# for the training
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--shuffle', type=str, default='trial', help='way to shuffle the data')
parser.add_argument('--logdir', type=str, default='logs', help='directory to save logs')
parser.add_argument('--checkpoint', type=int, default=1, help='save model after each checkpoint')
parser.add_argument('--multi_gpu', action='store_true', help='use multiple GPUs')
parser.add_argument('--verbose', type=int, default=1, help='print loss after each epoch')
# todo
parser.add_argument('--resume', type=str, default='', help='resume training from a checkpoint')

args = parser.parse_args()

logdir = os.path.join(args.logdir, f'{args.data}_{args.seed}')
if not os.path.exists(logdir):
    os.makedirs(logdir)


def main():
    # args = parser.parse_args()
    
    seed_everything(args.seed)
    
    X_train, _, _, y_train, _, _, _ = load_data(args.root, args.data, split=args.length)
    
    device = get_device()
    print(f'=> Running on {device}')
    
    model = MCP(
        input_dim=X_train.shape[-1],
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        length=args.length,
        depth=args.depth,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        momentum=args.momentum,
        queue_size=args.queue_size,
        num_queues=args.num_queues,
        multi_gpu=args.multi_gpu,
        callback_func=pretrain_callback
    )
    
    # train the model
    loss_list = model.fit(
        X_train,
        y_train,
        shuffle_function=args.shuffle,
        masks=args.masks,
        factors=args.factors,
        epochs=args.epochs,
        verbose=args.verbose
        )
    
    # save training loss
    np.save(os.path.join(logdir, 'loss.npy'), loss_list)
    
    
def pretrain_callback(model, epoch, checkpoint=args.checkpoint):
    if (epoch+1) % checkpoint == 0:
        model.save(os.path.join(logdir, 'weight', f'pretrain_{epoch+1}.pth'))
        

if __name__ == '__main__':
    main()


    
    
    
    
    
    



