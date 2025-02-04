# MCP: Momentum Contrast cross Patient

## Preparation

To install all required module, run one of the lines of code below:
```
pip install -r requirement.txt
conda install --file requirement.txt
```

## Data Preprocessing

### Chapman

1. Download the **ECGDataDenoised.zip** and **Diagnostics.xlsx** files from [official](https://figshare.com/collections/ChapmanECG/4560497/1)
2. Run the [chapman_preprocessing.ipynb](https://github.com/3hiuwoo/MCP/blob/main/data_preprocessing/chapman_preprocess.ipynb), remember to specify the path of raw data and destination for processed data.
3. Use the path of folder containing processed data to run following codes.

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

## Fine-tuning

To fine-tune with pretrained model, run:
```
python finetune.py  --data [dataset's name]\
--root [folder containing directories of datasets]\
--logdir [folder to save fine-tuned weights with best F1-score on validation set and loss]\
--pretrain [path of the pre-trained weight]
```
To change other default setting, run the following for details:
```
python finetune.py -h
```

## Testing

To test the fine-tuned model's performance, run:
```
python finetune.py  --data [dataset's name]\
--root [folder containing directories of datasets]\
--logdir [folder to save the testing outcome]\
--test [path to the fine-tuned weight]
```
Note that this will disable most of the training arguments.

## Baselines

For better alignment, we have implemented some baselines in ECG representation learning domain other than directly running their codes(some of them do not have), including [CLOCS](https://arxiv.org/abs/2005.13249), [ISL](https://arxiv.org/abs/2109.08908), see [MedDL](https://github.com/3hiuwoo/MedDL).

For [COMET](https://arxiv.org/abs/2310.14017), we pre-trained with their code, and fine-tuned with ours.

## Reference

We incorporate COMET [Code](https://github.com/DL4mHealth/COMET) to conduct aligned experiments.
