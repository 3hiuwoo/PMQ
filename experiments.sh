#!/bin/bash

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <dataset> <logdir> <is_pretrain>"
    exit 1
fi

datasets=(chapman ptb cpsc2018)
load_path=$1
logdir=$2

for dataset in ${datasets[@]}; do
    python finetune.py --data=$dataset --pretrain=$load_path --logdir=$logdir
done
