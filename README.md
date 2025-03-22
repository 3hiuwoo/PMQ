# MCP: Momentum Contrast cross Patient

## Requirements

Run the following to prepare all required modules.

```
pip install -r requirement.txt
# If you want to use conda
# conda install --file requirement.txt
```

## Datasets

### PTB-XL

### Chapman

1. Download the **ECGDataDenoised.zip** and **Diagnostics.xlsx** files from [here](https://figshare.com/collections/ChapmanECG/4560497/1) and extract the data from the .zip file.
2. Run the jupyter notebook [chapman_preprocessing.ipynb](https://github.com/3hiuwoo/MCP/blob/main/data_preprocessing/chapman_preprocess.ipynb) to preprocess the raw data, remember to modify the path in notebook about loading raw data and saving processed one.

### PTB

### PTB-XL

### training data organization
All processed data should be organized as below(all notebooks produce the data in this format automatically):

```
- [destination path specified in notebook]:
  - ptbxl:
    - features:
      - feature_00001.npy
      ...
    - labels:
      - labels.npy
  - other dataset
  ...
```

## Pre-training

To pre-train with the same setting as in the paper, just run:
```
python train.py --root [folder containing all datasets]\
--logdir [folder to save pre-trained model weights and training loss]
```

If you want to try different settings, run the following for details:

```
python train.py -h
```

## Fine-tuning

To fine-tune and test following our paper, run:

```
python finetune.py --root [folder containing all datasets]\
--logdir [folder to save fine-tuned weights and logs]\
--pretrain [path of the pre-trained weight file]

```
To fine-tune with any amount of datasets and any combinations of fractions with other settings, run the following for details:

```
python finetune.py -h
```

## Reference

We incorporate COMET's [Code](https://github.com/DL4mHealth/COMET) to conduct aligned experiments.
