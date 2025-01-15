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
2. Run the [jupyter notebook](https://github.com/3hiuwoo/MCP/blob/main/data_preprocessing/chapman_preprocess.ipynb), which allows you to select wherever you want to save the processed data folder, remember to modify the path to read unzipped original data in the notebook.
3. Use the path of folder containing processed dataset's folder to run following codes.

## Pretraining

To pre-train a temporal convolutional network with 10 residual blocks, run:
```
python train.py --data [dataset's name]\
--root [folder containing directories of datasets]\
--log [folder to save pretrained weight, loss]
```
To change other default setting, run the following for annotations for other arguments:
```
python train.py -h
```

## Linear Evaluation

To perform linear evaluation on pre-trained model, run:
```
python train.py --data [dataset's name]\
--root [folder containing directories of datasets]\
--log [folder to save linear evaluation outcome]\
--eval [path of the pretrained weight]
```
Note that this will disable most training arguments.

## Fine-tuning

To finetune with pretrained model, run:
```
python finetune.py  --data [dataset's name]\
--root [folder containing directories of datasets]\
--log [folder to save finetuned weights with best F1-score on validation set and loss]\
--pretrain [path of the pretrained weight]
```
To change other default setting, run the following for annotations for other arguments:
```
python finetune.py -h
```

## Testing

To test the finetuned model's performance, run:
```
python finetune.py  --data [dataset's name]\
--root [folder containing directories of datasets]\
--log [folder to save linear evaluation outcome]\
--test [path to the finetuned weight]
```
Note that this will disable most of the training arguments.

## Baselines

For better alignment, we implement some baselines in ECG representation learning domain other than directly running their codes(some of them do not have), including [CLOCS](https://arxiv.org/abs/2005.13249), [ISL](https://arxiv.org/abs/2109.08908), see [MedDL](https://github.com/3hiuwoo/MedDL).

For [COMET](https://arxiv.org/abs/2310.14017), we utilize their code in pretraining stage, and finetuned with our code.

## Reference

We incorporate COMET [Code](https://github.com/DL4mHealth/COMET) to perform aligned experiments.
