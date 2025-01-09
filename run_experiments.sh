#!/bin/bash

# This script runs experiments for several seeds, the pretrain and finetune dataset can be modified.
logdir=logs
pdataset=chapman
fdataset=chapman

for seed in {41..45}
do
    # comment out this line if the pretrained weight is already available
    python train.py --data=$pdataset --logdir=$logdir --seed=$seed

    python train.py --eval=${logdir}/mcp_${pdataset}_${seed}/pretrain_100.pth --data=$pdataset --logdir=$logdir --seed=$seed
    cat ${logdir}/mcp_${pdataset}_${seed}/log_${seed}.txt >> ${logdir}/log.txt

    python finetune.py --data=$fdataset --logdir=$logdir --seed=$seed --pretrain=${logdir}/mcp_${pdataset}_${seed}/pretrain_100.pth
    python finetune.py --fraction=0.1 --data=$fdataset --logdir=$logdir --seed=$seed --pretrain=${logdir}/mcp_${pdataset}_${seed}/pretrain_100.pth
    python finetune.py --fraction=0.01 --data=$fdataset --logdir=$logdir --seed=$seed --pretrain=${logdir}/mcp_${pdataset}_${seed}/pretrain_100.pth

    python finetune.py --test=${logdir}/fft_${fdataset}_${seed}/bestf1_1.pth --data=$fdataset --logdir=$logdir --seed=$seed
    python finetune.py --test=${logdir}/fft_${fdataset}_${seed}/bestf1_0.1.pth --fraction=0.1 --data=$fdataset --logdir=$logdir --seed=$seed
    python finetune.py --test=${logdir}/fft_${fdataset}_${seed}/bestf1_0.01.pth --fraction=0.01 --data=$fdataset --logdir=$logdir --seed=$seed
    cat ${logdir}/fft_${fdataset}_${seed}/log_${seed}.txt >> ${logdir}/log.txt
    echo "" >> ${logdir}/log.txt
done