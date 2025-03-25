# MCP: Momentum Contrast cross Patient

## Requirements

Run the following to prepare all required modules.

```
pip install -r requirement.txt
# If you want to use conda
# conda install --file requirement.txt
```

## Datasets

### Download

- **PTB-XL**: Download the zip file from [here](https://physionet.org/content/ptb-xl/1.0.3/) and extract the data.

- **Chapman**: Download the *ECGDataDenoised.zip* and *Diagnostics.xlsx* files from [here](https://figshare.com/collections/ChapmanECG/4560497/1) and extract the data from the .zip file.

- **PTB**: Download the zip file from [here](https://physionet.org/content/ptbdb/1.0.0/) and extract the data.

- **CPSC2018**: Download the zip file from [here](https://www.kaggle.com/datasets/bjoernjostein/china-12lead-ecg-challenge-database) and extract the data.

### Preprocessing

Run jupyter notebooks corresponding to each dataset from this [folder](https://github.com/3hiuwoo/MCP/blob/main/data_preprocessing) to preprocess the raw data, remember to modify the path in notebooks to load your downloaded dataset and to save processed dataset.

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

The "destination path specified in notebook" will be used by all scripts to load data.

## Pre-training

To pre-train with the same setting as in the paper, just run:
```
python train.py --root [folder containing all datasets]\
                --logdir [folder to save weights and training loss]\
                --neighbor\
                --use_id
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
