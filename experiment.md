# 2024-11-15

Pretraining using **CMSC** on **Chapman** and finetuning on **CINC2017** using 12 leads and an learned embedding dimension of 256, such setting is which performs the best in the original paper:

|            | AUC      | Accuracy | F1score  |
|:-----------|---------:|---------:|---------:|
| Supervised | 0.860281 | 0.670341 | 0.589147 |
| Finetune   | 0.874503 | 0.694389 | 0.622155 |

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