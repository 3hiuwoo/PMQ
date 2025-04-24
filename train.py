"""
Pretrain with PMQ or PMB.
"""
import argparse
import os
import warnings

import numpy as np

from data import load_data
from model import PMB, PMQ
from utils import get_device, seed_everything

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="PMQ training")
parser.add_argument("--seed", type=int, default=42, help="random seed")
# data
parser.add_argument("--root", type=str, default="/root/autodl-tmp/dataset", help="root directory of datasets")
parser.add_argument("--data", type=str, default="ptbxl", help="pretraining dataset: [chapman, ptb, ptbxl, cpsc2018]")
parser.add_argument("--length", type=int, default=600, help="length of each sample")
parser.add_argument("--overlap", type=float, default=0., help="overlap of each sample")
parser.add_argument("--neighbor", action="store_true", help="whether to segment sample into neighbors")
# model
parser.add_argument("--depth", type=int, default=10, help="number of hidden dilated convolutional blocks")
parser.add_argument("--hidden_dim", type=int, default=64, help="output dimension of input projector")
parser.add_argument("--output_dim", type=int, default=320, help="output dimension of input projector")
parser.add_argument("--momentum", type=float, default=0.999, help="momentum update parameter")
parser.add_argument("--tau", type=float, default=1.0, help="temperature for cosine similarity")
parser.add_argument("--alpha", type=float, default=2.0, help="scaling factor of positive pairs in multi-similarity loss")
parser.add_argument("--beta", type=float, default=50.0, help="scaling factor of negative pairs in multi-similarity loss")
parser.add_argument("--thresh", type=float, default=0.5, help="bias for pairs in multi-similarity loss")
parser.add_argument("--margin", type=float, default=0.1, help="margin for mining pairs in multi-similarity loss")
parser.add_argument("--mask_t", type=float, default=0.5, help="probability of time mask")
parser.add_argument("--mask_f", type=float, default=0.1, help="ratio of freq mask")
parser.add_argument("--pool", type=str, default="avg", help="pooling method for representation: [avg, max]")
parser.add_argument("--queue_size", type=int, default=16384, help="queue size for TFPQ, set 0 for TFPB")
parser.add_argument("--use_id", action="store_true", help="whether to use patient-infoNCE loss")
parser.add_argument("--loss_func", type=str, default="nce", help="contrastive loss function (add a/s to the end to compute loss (a)symmetrically in PMB): [ms, nce](a/s)")
# training
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
parser.add_argument("--optim", type=str, default="adamw", help="optimizer: [adamw, adam, lars]")
parser.add_argument("--schedule", type=str, default=None, help="scheduler: [plateau, step, cosine, warmup, exp]")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--shuffle", type=str, default="trial", help="way to shuffle the data: [random, trial]")
parser.add_argument("--logdir", type=str, default="log", help="directory to save weights and logs")
parser.add_argument("--checkpoint", type=int, default=1, help="frequency to save checkpoint")
parser.add_argument("--multi_gpu", action="store_true", help="whether to use multiple GPUs")
parser.add_argument("--verbose", type=int, default=1, help="if large than 0: print loss after each epoch")

def main(): 
    args = parser.parse_args()
    print("=> Arguments:", vars(args))
    
    logdir = os.path.join(args.logdir, f"pretrain_{args.data}_{args.seed}")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
        
    print(f"=> Weights will be saved in {logdir}")
    
    # save argumens information
    with open(os.path.join(logdir, "args.txt"), "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        
    seed_everything(args.seed)
    print(f"=> Set seed to {args.seed}")
    
    X_train, X_val, X_test,\
    y_train, y_val, y_test = load_data(args.root,
                                       args.data,
                                       length=args.length,
                                       overlap=args.overlap,
                                       neighbor=args.neighbor
                                       )
    
    device = get_device()
    print(f"=> Running on {device}")
    
    if args.queue_size:
        model = PMQ(
            input_dims=X_train.shape[-1],
            output_dims=args.output_dim,
            hidden_dims=args.hidden_dim,
            depth=args.depth,
            pool=args.pool,
            mask_t=args.mask_t,
            mask_f=args.mask_f,
            momentum=args.momentum,
            tau=args.tau,
            alpha=args.alpha,
            beta=args.beta,
            thresh=args.thresh,
            margin=args.margin,
            queue_size=args.queue_size,
            use_id=args.use_id,
            loss_func=args.loss_func,
            device=device,
            multi_gpu=args.multi_gpu
        )
    else:
        print(f"=> !!! Training without queue !!!")
        model = PMB(
            input_dims=X_train.shape[-1],
            output_dims=args.output_dim,
            hidden_dims=args.hidden_dim,
            depth=args.depth,
            pool=args.pool,
            mask_t=args.mask_t,
            mask_f=args.mask_f,
            momentum=args.momentum,
            tau=args.tau,
            alpha=args.alpha,
            beta=args.beta,
            thresh=args.thresh,
            margin=args.margin,
            use_id=args.use_id,
            loss_func=args.loss_func,
            device=device,
            multi_gpu=args.multi_gpu
        )
        
    loss_list = model.fit(
        X_train,
        y_train,
        shuffle_function=args.shuffle,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        optim=args.optim,
        schedule=args.schedule,
        logdir=logdir,
        checkpoint=args.checkpoint,
        verbose=args.verbose
        )
    
    np.save(os.path.join(logdir, "loss.npy"), loss_list)

if __name__ == "__main__":
    main()


    
    
    
    
    
    



