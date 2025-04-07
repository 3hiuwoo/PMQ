# CLOCS Unofficial Implementation

We implement the CMSC variant from CLOCS following the official implementation in order to better align with our experiments.

## Preparation

Following the global setting as [PMQ](https://github.com/3hiuwoo/PMQ/blob/main/README.md)

## Pre-training

To pre-train with the same setting as in the paper, just run:

```shell
python train.py --root [folder containing all datasets]\
                --logdir [folder to save weights and training loss]\
```

If you want to try different settings, run the following for details:

```shell
python train.py -h
```

## Fine-tuning

To fine-tune and test following our paper, run:

```shell
python finetune.py --root [folder containing all datasets]\
                   --logdir [folder to save fine-tuned weights and logs]\
                   --pretrain [path of the pre-trained weight file]
```

To fine-tune with any amount of datasets and any combinations of fractions with other settings, run the following for details:

```shell
python finetune.py -h
```

## Reference

1. CLOCS [paper](https://proceedings.mlr.press/v139/kiyasseh21a/kiyasseh21a.pdf)
2. CLOCS [repo](https://github.com/danikiyasseh/CLOCS.git)


