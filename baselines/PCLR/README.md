# PCLR Unofficial Implementation

We implement the PCLR following the paper in order to better align with our experiments.

## Preparation

Following the global setting as [PMQ](https://github.com/3hiuwoo/PMQ/blob/main/README.md).

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

See [finetune.py](https://github.com/3hiuwoo/PMQ?tab=readme-ov-file#fine-tuning)

## Reference

1. PCLR [paper](https://arxiv.org/abs/2104.04569)
