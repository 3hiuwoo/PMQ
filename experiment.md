# EXP1: Compare with supervised

Pretrain with **CMSC** on **Chapman** and finetuning on **CINC2017** using 12 leads and an learned embedding dimension of 256, such setting is which performs the best in the original paper:

|              | AUC      | Accuracy | F1score  |
|:-------------|---------:|---------:|---------:|
| Supervised   | 0.860281 | 0.670341 | 0.589147 |
| Finetune w/  | 0.874503 | 0.694389 | 0.622155 |
| Finetune w/o | 0.873064 | 0.681363 | 0.598361 |

There has been an update on Chapman dataset enlarging its shape to more than 40k 12-lead ecg frames(from 10k), boosting the CMSC method from auc of 0.822 to one listed above. And the supervised model trained from scratch performs not as poor as shown in the paper.

About the dataset:
**The following findings are possibly related to my technical problem or misunderstanding**
- a small portion of data has labels that do not belong to any class out of 4 classes described by the dataset introduction, we dropped them.
- a small portion of data has labels corresponding to not only one class out of 4 classes, we dropped them.
- a small portion of data has leads that full of 0 value or contains some NaN value, we dropped them.

What else can we do:
- apply **denoising** on **Chapman**
- apply more other transformations
- **linear evaluation**
- test different bigger size models
- compare pretrained model with supervised one with less available labeld data
- try more contrastive learning method(MoCo)
- try bigger merged pretrain dataset
- test on more downstream dataset
- run more baseline(CMLC, CMSMLC, etc)
- try more batch size
- ablation on loss
- try projection head
- apply new loss to moco
- change the fc

# EXP2: Ablation on batch size and epoch

Pretrain with **CMSC** on **Chapman** and finetuning on **CINC2017** using 12 leads and an learned embedding dimension of 256, but with different batch sizes:

## pretrain for 100 epochs

| batch size | AUC      | Accuracy | F1score  |
|:-----------|---------:|---------:|---------:|
| 256        | 0.869519 | 0.697395 | 0.620368 |
| 2048       | 0.881905 | 0.691383 | 0.623343 |

## pretrain for 400 epochs

| batch size | AUC          | Accuracy     | F1score      |
|:-----------|-------------:|-------------:|-------------:|
| 64         | 0.882743     | 0.707415     | **0.638528** |
| 128        | 0.878578     | **0.708417** | 0.629871     |
| 256        | 0.874503     | 0.694389     | 0.622155     |
| 512        | 0.866470     | 0.688377     | 0.621465     |
| 1024       | 0.881802     | 0.698397     | 0.618134     |
| 2048       | **0.883140** | 0.695391     | 0.617683     |
| 4096       | 0.880481     | 0.687375     | 0.601843     |

# EXP3: Compare with SimCLR

## pretrain for 100 epochs

| batch size | AUC      | Accuracy | F1score  |
|:-----------|---------:|---------:|---------:|
| 256        | 0.869519 | 0.697395 | 0.620368 |
| 2048       | 0.881905 | 0.691383 | 0.623343 |

## pretrain for 400 epochs

| batch size | AUC      | Accuracy | F1score  |
|:-----------|---------:|---------:|---------:|
| 256        | 0.869480 | 0.689379 | 0.613317 |
| 4086       | 0.860255 | 0.673347 | 0.592742 |