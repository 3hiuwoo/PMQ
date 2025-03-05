# MCP: Momentum Contrast cross Patient

## Preparation

To install all required modules, run one of the lines of code below:
```
pip install -r requirement.txt
conda install --file requirement.txt
```

## Data Preprocessing

### Chapman

1. Download the **ECGDataDenoised.zip** and **Diagnostics.xlsx** files from [official](https://figshare.com/collections/ChapmanECG/4560497/1) and extract the data from the .zip file.
2. Run the jupyter notebook [chapman_preprocessing.ipynb](https://github.com/3hiuwoo/MCP/blob/main/data_preprocessing/chapman_preprocess.ipynb) to preprocess the raw data, remember to specify the path of raw data and for processed one.

### PTB

**Debug**: waiting for fill

### PTB-XL

**Debug**: waiting for fill

### training data organization
All processed data should be organized as below(all notebooks produce the data in this format automatically):
```
- [specified path in notebook]:
  - chapman:
    - features:
      - feature_00001.npy
      ...
    - labels:
      - labels.npy
  - other dataset
  ...
```

## Pre-training

To pre-train a temporal convolutional network with 10 residual blocks, run:
```
python train.py --data [dataset's name]\
--root [folder containing directories of datasets]\
--logdir [folder to save pretrained weight, loss]
```
To change other default settings, run the following for details:
```
python train.py -h
```
**Debug**: train2.py train3.py, etc. are for experimental test, which share the same arguments with train.py, see EXP.md for more details.

## Fine-tuning

To fine-tune and test with pretrained model, run:
```
python finetune.py  --data [dataset's name]\
--root [folder containing directories of datasets]\
--logdir [folder to save fine-tuned weights with best F1-score on validation set and loss]\
--pretrain [path of the pre-trained weight file]
```
To change other default setting, run the following for details:
```
python finetune.py -h
```
**Debug**: finetune2.py is for fine-tuning the backbone modified in train2.py.

## Baselines

We implement [CLOCS](https://arxiv.org/abs/2005.13249) for alignment, see [MedDL](https://github.com/3hiuwoo/MedDL).

For other baselines, we utilize their code repositories to pre-train the model, then fine-tuned with ours. Those baselines include: [COMET](https://arxiv.org/abs/2310.14017), [TimeSiam](https://arxiv.org/abs/2402.02475), [TFC](https://arxiv.org/abs/2206.08496), [TS2Vec](https://arxiv.org/abs/2106.10466).


## Reference

We incorporate COMET [Code](https://github.com/DL4mHealth/COMET) to conduct aligned experiments.
