# 2025/1/4

## 4090, batch: 256, queue: 4096, lr: 1e-4, pretrain-epochs:100, finetune-epochs:50

### factors: [0.25, 0.25, 0.25, 0.25]

#### Linear Evaluation

| Model     | AUROC     | Accuracy  | F1score   | Precision | Recall    | AUPRC     |
|:----------|----------:|----------:|----------:|----------:|----------:|----------:|
| MCP       | 0.854675  | 0.576072  | 0.582659  | 0.634560  | 0.603268  | 0.676467  |
| MPF       | 0.881501  | 0.595636  | 0.658062  | 0.705602  | 0.695628  | 0.748562  |
| COMET     | 0.896095  | 0.650865  | 0.709050  | 0.734566  | 0.731920  | 0.771784  |
| CMSC      | ########  | ########  | ########  | ########  | ########  | #######   |
| ISL       | ########  | ########  | ########  | ########  | ########  | #######   |

#### Full Fine-Tune

##### faction: 1.0

| Model     | AUROC     | Accuracy  | F1score   | Precision | Recall    | AUPRC     |
|:----------|----------:|----------:|----------:|----------:|----------:|----------:|
| Scratch   | 0.906144  | 0.727765  | 0.767283  | 0.768488  | 0.767514  | 0.805540  |
| COMET+    | 0.926493  | 0.771633  | 0.782538  | 0.802594  | 0.785412  | 0.849381  |
| Time-Freq | 0.913006  | 0.753123  | 0.775886  | 0.779683  | 0.778660  | 0.818178  |
| COMET     | 0.926532  | 0.775814  | 0.768394  | 0.795910  | 0.784722  | 0.842812  |
| CMSC      | 0.898110  | 0.637698  | 0.714278  | 0.745486  | 0.746277  | 0.785959  |
| CMSC+     | 0.913992  | 0.732280  | 0.767492  | 0.762337  | 0.773231  | 0.817240  |
| ISL       | ########  | ########  | ########  | ########  | ########  | #######   |

##### faction: 0.1

| Model     | AUROC     | Accuracy  | F1score   | Precision | Recall    | AUPRC     |
|:----------|----------:|----------:|----------:|----------:|----------:|----------:|
| Scratch   | 0.912111  | 0.682318  | 0.740574  | 0.768945  | 0.753417  | 0.816443  |
| COMET+    | 0.933911  | 0.772385  | 0.765852  | 0.775883  | 0.778137  | 0.845581  |
| Time-Freq | 0.907732  | 0.615425  | 0.682379  | 0.746788  | 0.722370  | 0.797198  |
| COMET     | 0.898944  | 0.646125  | 0.713960  | 0.733068  | 0.740235  | 0.776362  |
| CMSC      | 0.887007  | 0.606320  | 0.679563  | 0.746446  | 0.733217  | 0.761602  |
| CMSC+     | 0.912111  | 0.682318  | 0.740575  | 0.768945  | 0.753417  | 0.816443  |
| ISL       | ########  | ########  | ########  | ########  | ########  | #######   |

##### faction: 0.01

| Model     | AUROC     | Accuracy  | F1score   | Precision | Recall    | AUPRC     |
|:----------|----------:|----------:|----------:|----------:|----------:|----------:|
| Scratch   | 0.875684  | 0.631678  | 0.648663  | 0.672914  | 0.662661  | 0.714983  |
| COMET+    | 0.925928  | 0.755455  | 0.783427  | 0.787084  | 0.787112  | 0.824541  |
| Time-Freq | 0.916155  | 0.714296  | 0.755284  | 0.773086  | 0.757394  | 0.814523  |
| COMET     | 0.859955  | 0.666290  | 0.688875  | 0.680563  | 0.700259  | 0.720004  |
| CMSC      | 0.855563  | 0.543115  | 0.621058  | 0.670388  | 0.672959  | 0.713244  |
| CMSC+     | 0.875685  | 0.631678  | 0.648663  | 0.672914  | 0.662661  | 0.714983  |
| ISL       | ########  | ########  | ########  | ########  | ########  | #######   |

Experiments conducted so far:
- **COMET** 1seed **1run** (../comet/test_run/chapman/0.25.../logs) 2run (log_comet)
- COMET drop_last 1seed (log_comet)
- **COMET w momentum** 1seed (log_mcp)
- COMET w momentum debug 1seed (log_mcp2)
- **Time-Frequency w Momentum trial shuffle** 1seed (log_mpf)
- time-frequency w momentum random shuffle 1seed (log_mpfr)
- **CMSC** 1seed (log_cmsc)
- **CMSC w momentum** 1seed (log_mcl)

# 2025/1/25

## 4090 batch256 pretrain 100 epochs finetune 50 epochs lr1e-4 queuesize16384

#### Chapman

##### faction: 1.0

| Model     | AUROC     | Accuracy  | F1score   | Precision | Recall    | AUPRC     |
|:----------|----------:|----------:|----------:|----------:|----------:|----------:|
| RandInit  | 0.906144  | 0.727765  | 0.767283  | 0.768488  | 0.767514  | 0.805540  |
| TFP       | 0.913006  | 0.753123  | 0.775886  | 0.779683  | 0.778660  | 0.818178  |
| COMET     | 0.926532  | 0.775814  | 0.768394  | 0.795910  | 0.784722  | 0.842812  |
| CLOCS     | 0.898110  | 0.637698  | 0.714278  | 0.745486  | 0.746277  | 0.785959  |
| TFC       | ########  | ########  | ########  | ########  | ########  | #######   |
| TS2Vec    | ########  | ########  | ########  | ########  | ########  | #######   |

##### faction: 0.1

| Model     | AUROC     | Accuracy  | F1score   | Precision | Recall    | AUPRC     |
|:----------|----------:|----------:|----------:|----------:|----------:|----------:|
| RandInit  | 0.906144  | 0.727765  | 0.767283  | 0.768488  | 0.767514  | 0.805540  |
| TFP       | 0.913006  | 0.753123  | 0.775886  | 0.779683  | 0.778660  | 0.818178  |
| COMET     | 0.926532  | 0.775814  | 0.768394  | 0.795910  | 0.784722  | 0.842812  |
| CLOCS     | 0.898110  | 0.637698  | 0.714278  | 0.745486  | 0.746277  | 0.785959  |
| TFC       | ########  | ########  | ########  | ########  | ########  | #######   |
| TS2Vec    | ########  | ########  | ########  | ########  | ########  | #######   |

##### faction: 0.01

| Model     | AUROC     | Accuracy  | F1score   | Precision | Recall    | AUPRC     |
|:----------|----------:|----------:|----------:|----------:|----------:|----------:|
| RandInit  | 0.906144  | 0.727765  | 0.767283  | 0.768488  | 0.767514  | 0.805540  |
| TFP       | 0.913006  | 0.753123  | 0.775886  | 0.779683  | 0.778660  | 0.818178  |
| COMET     | 0.926532  | 0.775814  | 0.768394  | 0.795910  | 0.784722  | 0.842812  |
| CLOCS     | 0.898110  | 0.637698  | 0.714278  | 0.745486  | 0.746277  | 0.785959  |
| TFC       | ########  | ########  | ########  | ########  | ########  | #######   |
| TS2Vec    | ########  | ########  | ########  | ########  | ########  | #######   |

# 2025/2/9

## Chapman

### 100%

| Method   | F1              | AUROC           | ACC             |
|----------|-----------------|-----------------|-----------------|
| Ours     |0.737977±0.018080|0.915647±0.008716|0.688518±0.045793|
| Ours t   |0.743887±0.022200|0.903783±0.013904|0.689917±0.033115|
| COMET    |0.735831±0.044075|0.904718±0.017663|0.690534±0.077640|
| CMSC     |0.683985±0.049822|0.864386±0.041261|0.619398±0.055824|
| TS2Vec   |0.730068±0.038761|0.899746±0.013012|0.666746±0.051525｜
| TFC      |0.652935±0.021653|0.865490±0.004662|0.582408±0.022004|
| TimeSiam |0.720110±0.046994|0.888026±0.021104|0.659970±0.058050|
| RandInit |0.736630±0.022066|0.908122±0.013946|0.680527±0.036283|

### 10%

| Method   | F1              | AUROC           | ACC             |
|----------|-----------------|-----------------|-----------------|
| Ours     |0.722632±0.008494|0.907059±0.003953|0.696644±0.008710|
| Ours t   |0.717783±0.015382|0.901555±0.002945|0.663537±0.020738|
| COMET    |0.714932±0.035560|0.896690±0.014024|0.666065±0.065775|
| CMSC     |0.641599±0.010525|0.872117±0.015033|0.570625±0.007935|
| TS2Vec   |0.753921±0.031675|0.921356±0.008886|0.713318±0.056172|
| TFC      |0.625803±0.005685|0.842581±0.002065|0.603642±0.010206|
| TimeSiam |0.707387±0.019069|0.900516±0.010599|0.642739±0.026978|
| RandInit |0.707464±0.012162|0.892010±0.005545|0.655907±0.012788|

### 1%

| Method   | F1              | AUROC           | ACC             |
|----------|-----------------|-----------------|-----------------|
| Ours     |0.443722±0.022707|0.762842±0.012186|0.439774±0.021819|
| Ours t   |0.393832±0.012810|0.714108±0.010134|0.404379±0.011404|
| COMET    |0.728935±0.028130|0.891320±0.013446|0.721354±0.023617|
| CMSC     |0.647798±0.027875|0.866333±0.035192|0.575139±0.029201|
| TS2Vec   |0.769610±0.011514|0.924092±0.004507|0.729842±0.018604|
| TFC      |0.590602±0.007333|0.816873±0.002435|0.613409±0.003725|
| TimeSiam |0.733125±0.029907|0.904304±0.011181|0.698224±0.042799|
| RandInit |0.365189±0.021937|0.695228±0.011683|0.380573±0.020181|

### Summary

At first, we trained the model with directly feeding time and frequency spectrum to each branch respectively, but the result (**2025/1/25**) cannot validate the model because it showed catastrophical performance deprecation under data scarcity situation, which is controversy to the expectation of a well pre-trained model.

With the assumption that it can confuse the model feeding both temporal and spectral data to it with different contextual meaning, we expand the single projector in front of the CNN to two, receiving data in two space respectively (**train2.py**), in a sense that this will project two set of data in different space into a same new space, and we add the two projected features together to feed the CNN. now each branch receive two set of input while pre-training, temporal and spectral.

When fine-tuning, we can generate the spectral of the input batch and utilizing two input layers (**finetune2.py**), or we can drop the input layer for spectral data and fine-tune like before (**finetune.py**), these method corresponds to 'ours' and 'ours t' in the above table. 

And we also tried using two independent encoder for pre-training(**train3.py**), dropping the momentum encoder, also avoid the problem that the same encoder receives both temporal and spectral data.

Unfortunately, this round of trying finally fell into almost complete failure. Although the two input projector model shows slightly better performance than baselines under full data scenario, but its performance consistently drops significantly as the data fraction decreases. Even worse, the model trained from scratch under supervised learning shows similar performance to our method, proving that this design cannot learn a robust enough representation that tunes the model to a more favorable feature space.

A new assumption is that the addition of two projected feature still prevents the model from learning pattern in ECG.


