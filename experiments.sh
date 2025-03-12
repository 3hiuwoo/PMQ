#!/bin/bash

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <dataset> <logdir> <is_pretrain>"
    exit 1
fi

datasets=(ptb cpsc2018 cinc2017)
load_path=$1
logdir=$2
is_pretrain=$3

if [ $is_pretrain -eq 1 ]; then
    python train_tfp.py --logdir=$logdir --lr=1e-5 --schedule=exp --data=ptbxl
else
    for dataset in ${datasets[@]}; do
        python finetune.py --data=$dataset --pretrain=$load_path --logdir=$logdir
    done
fi