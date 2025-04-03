import os
import sys
import argparse
import warnings

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from utils import get_device, seed_everything
from pclr import PCLR, ECGDataset

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="PCLR training")
parser.add_argument("--seed", type=int, default=42, help="random seed")
# data
parser.add_argument("--root", type=str, default="/root/autodl-tmp/dataset", help="root directory of datasets")
parser.add_argument("--data", type=str, default="ptbxl", help="pretraining dataset: [chapman, ptb, ptbxl]")
parser.add_argument("--length", type=int, default=300, help="length of each sample")
parser.add_argument("--overlap", type=float, default=0., help="overlap of each sample")
# model
parser.add_argument("--depth", type=int, default=10, help="number of hidden dilated convolutional blocks")
parser.add_argument("--hidden_dim", type=int, default=64, help="output dimension of input projector")
parser.add_argument("--output_dim", type=int, default=320, help="output dimension of input projector")
parser.add_argument("--tau", type=float, default=0.1, help="temperature for cosine similarity")
# training
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--wd", type=float, default=1.5e-6, help="weight decay")
parser.add_argument("--optim", type=str, default="adamw", help="optimizer: [adamw, lars]")
parser.add_argument("--schedule", type=str, default=None, help="scheduler: [plateau, step, cosine, warmup, exp]")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
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
    
    dataset = ECGDataset(args.root,
                         args.data,
                         length=args.length,
                         overlap=args.overlap
                         )
    
    device = get_device()
    print(f"=> Running on {device}")
    
    print(f"=> Training PCLR")
    model = PCLR(
        input_dims=dataset.X_train.shape[-1],
        output_dims=args.output_dim,
        hidden_dims=args.hidden_dim,
        depth=args.depth,
        tau=args.tau,
        device=device,
        multi_gpu=args.multi_gpu
    )
    
    loss_list = model.fit(
        train_dataset=dataset,
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


    
    
    
    
    
    



