# Create BERTVision Environment

```
conda create -n my_ml python=3.7 pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
conda activate my_ml
conda install -c conda-forge jupyter numpy pandas matplotlib scikit-learn pip python-language-server[all] tensorflow-gpu transformers datasets elasticsearch tqdm
conda install -c install tokenizers
pip install loguru hyperopt
```


# Hyperparameter Searching

We use the following hyperparameters drawing from our own `hyperopt` hyperparameter searches and academic testing.
We find that smaller data sets are much more sensitive to the learning rate than larger ones.

Our hyperparameter search can be run with the following command:

```
python -m models.hypersearch --model MSR --checkpoint bert-large-uncased --batch-size 32 --num-labels 2 --max-seq-length 128
python -m models.ap_hypersearch --model AP_STSB --checkpoint bert-base-uncased --batch-size 16 --num-labels 1 --max-seq-length 128
```

For large data sets, e.g., MNLI, QNLI, and QQP, data set sharding is enabled automatically, which
randomly samples 15% of the data set to train on to speed up the search.

Ensure that you specify the GLUE task `model` as well as the appropriate values for `batch-size`, `num-labels` and `max-seq-length`.

BERT-Large | MNLI | QNLI | QQP | RTE | SST | MSR | CoLA | STS-B
---|---|---|---|---|---|---|---|---
`--num-labels` | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 1
`--lr` | 1e-5 | 1e-5 | 1e-5 | 2e-5 | 2e-5 | 2e-5 | 2e-5 | 2e-5
`--batch-size` | 32 | 32 | 32 | 16 | 32 | 32 | 16 | 16
`--max-seq-length` | 128 | 128 | 128 | 250 | 128 | 128 | 128 | 128

BERT-Base | MNLI | QNLI | QQP | RTE | SST | MSR | CoLA | STS-B
---|---|---|---|---|---|---|---|---
`--num-labels` | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 1
`--lr` | 1e-5 | 1e-5 | 1e-5 | 2e-5 | 2e-5| 2e-5 | 2e-5 | 2e-5
`--batch-size` | 32 | 32 | 32 | 16 | 32 | 32 | 16 | 16
`--max-seq-length` | 128 | 128 | 128 | 250 | 128 | 128 | 128 | 128

# Our Results
BERT-base and AP Models

<img align="center" src="../../images/bert_base_vs_bertvision.png" />


# Latest Results

## GLUE

### QQP

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```
python -m models.bert_glue --model QQP --checkpoint bert-base-uncased --lr 1e-5 --num-labels 2 --max-seq-length 128 --batch-size 32
python -m models.ap_glue --model AP_QQP --checkpoint bert-base-uncased --lr 1e-5 --num-labels 2 --max-seq-length 128 --batch-size 32

python -m models.bert_glue --model QQP --checkpoint bert-large-uncased --lr 1e-5 --num-labels 2 --max-seq-length 128 --batch-size 32
python -m models.ap_glue --model AP_QQP --checkpoint bert-large-uncased --lr 1e-5 --num-labels 2 --max-seq-length 128 --batch-size 32
```

|BERT-base QQP | BERT-large QQP | BERTVision-base QQP |  BERTVision-large QQP |
|--|--|--|--|
|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.889</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.869</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.886</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.897</td></tr></table>|

### QNLI

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```
python -m models.bert_glue --model QNLI --checkpoint bert-base-uncased --lr 1e-5 --num-labels 2 --max-seq-length 128 --batch-size 32
python -m models.ap_glue --model AP_QNLI --checkpoint bert-base-uncased --lr 2.0021e-5 --num-labels 2 --max-seq-length 128 --batch-size 32 --seed 770

python -m models.bert_glue --model QNLI --checkpoint bert-large-uncased --lr 1e-5 --num-labels 2 --max-seq-length 128 --batch-size 32
python -m models.ap_glue --model AP_QNLI --checkpoint bert-large-uncased --lr 1e-5 --num-labels 2 --max-seq-length 128 --batch-size 32
```

|BERT-base QNLI | BERT-large QNLI | BERTVision-base QNLI |  BERTVision-large QNLI |
|--|--|--|--|
|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.901</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.912</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.892</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.910</td></tr></table>|

### MNLI

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```
python -m models.bert_glue --model MNLI --checkpoint bert-base-uncased --lr 1e-5 --num-labels 3 --max-seq-length 128 --batch-size 32
python -m models.ap_glue --model AP_MNLI --checkpoint bert-base-uncased --lr 1e-5 --num-labels 3 --max-seq-length 128 --batch-size 32

python -m models.bert_glue --model MNLI --checkpoint bert-large-uncased --lr 1e-5 --num-labels 3 --max-seq-length 128 --batch-size 32
python -m models.ap_glue --model AP_MNLI --checkpoint bert-large-uncased --lr 1e-5 --num-labels 3 --max-seq-length 128 --batch-size 32
```

|BERT-base MNLI | BERT-large MNLI | BERTVision-base MNLI |  BERTVision-large MNLI |
|--|--|--|--|
|<table><tr><th>Dev. Matched Acc.</th><th>Dev. Mismatched Acc.</th></tr><tr><td>0.824</td><td>0.829</td></tr></table>|<table><tr><th>Dev. Matched Acc.</th><th>Dev. Mismatched Acc.</th></tr><tr><td>0.853</td><td>0.851</td></tr></table>|<table><tr><th>Dev. Matched Acc.</th><th>Dev. Mismatched Acc.</th></tr><tr><td>0.822</td><td>0.829</td></tr></table>|<table><tr><th>Dev. Matched Acc.</th><th>Dev. Mismatched Acc.</th></tr><tr><td>0.849</td><td>0.850</td></tr></table>|



### RTE

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```
python -m models.bert_glue --model RTE --checkpoint bert-base-uncased --lr 1.2220e-5 --num-labels 2 --max-seq-length 250 --batch-size 16 --seed 600
python -m models.ap_glue --model AP_RTE --checkpoint bert-base-uncased --lr 1.64116e-5 --num-labels 2 --max-seq-length 250 --batch-size 16 --seed 41

python -m models.bert_glue --model RTE --checkpoint bert-large-uncased --lr 8.3621e-6 --num-labels 2 --max-seq-length 250 --batch-size 16 --seed 244
python -m models.ap_glue --model AP_RTE --checkpoint bert-large-uncased --lr 1.2614e-5 --num-labels 2 --max-seq-length 250 --batch-size 16 --seed 414
```

|BERT-base RTE | BERT-large RTE | BERTVision-base RTE |  BERTVision-large RTE |
|--|--|--|--|
|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.657</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.664</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.661</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.592</td></tr></table>|


### SST

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```
python -m models.bert_glue --model SST --checkpoint bert-base-uncased --lr 2e-5 --num-labels 2 --max-seq-length 128 --batch-size 32
python -m models.ap_glue --model AP_SST --checkpoint bert-base-uncased --lr 2e-5 --num-labels 2 --max-seq-length 128 --batch-size 32

python -m models.bert_glue --model SST --checkpoint bert-large-uncased --lr 2e-5 --num-labels 2 --max-seq-length 128 --batch-size 32
python -m models.ap_glue --model AP_SST --checkpoint bert-large-uncased --lr 2e-5 --num-labels 2 --max-seq-length 128 --batch-size 32
```

|BERT-base SST | BERT-large SST | BERTVision-base SST |  BERTVision-large SST |
|--|--|--|--|
|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.920</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.933</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.922</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.929</td></tr></table>|

### MSR

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```
python -m models.bert_glue --model MSR --checkpoint bert-base-uncased --lr 2.4380e-5 --num-labels 2 --max-seq-length 128 --batch-size 32 --seed 734
python -m models.ap_glue --model AP_MSR --checkpoint bert-base-uncased --lr 2.7181e-5 --num-labels 2 --max-seq-length 128 --batch-size 32 --seed 760

python -m models.bert_glue --model MSR --checkpoint bert-large-uncased --lr 1.2771e-5 --num-labels 2 --max-seq-length 128 --batch-size 32 --seed 158
python -m models.ap_glue --model AP_MSR --checkpoint bert-large-uncased --lr 1.4399e-5 --num-labels 2 --max-seq-length 128 --batch-size 32 --seed 465
```

|BERT-base MSR | BERT-large MSR | BERTVision-base MSR |  BERTVision-large MSR |
|--|--|--|--|
|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.828</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.768</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.802</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.837</td></tr></table>|

### CoLA

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```
python -m models.bert_glue --model CoLA --checkpoint bert-base-uncased --lr 9.6296e-6 --num-labels 2 --max-seq-length 128 --batch-size 16 --seed 441
python -m models.ap_glue --model AP_CoLA --checkpoint bert-base-uncased --lr 2.25972e-5 --num-labels 2 --max-seq-length 128 --batch-size 16 --seed 563

python -m models.bert_glue --model CoLA --checkpoint bert-large-uncased --lr 9.4471e-6 --num-labels 2 --max-seq-length 128 --batch-size 16 --seed 203
python -m models.ap_glue --model AP_CoLA --checkpoint bert-large-uncased --lr 2.99619e-5 --num-labels 2 --max-seq-length 128 --batch-size 16 --seed 949
```

|BERT-base CoLA | BERT-large CoLA | BERTVision-base CoLA |  BERTVision-large CoLA |
|--|--|--|--|
|<table><tr><th>Dev. Matthews</th></tr><tr><td>0.565</td></tr></table>|<table><tr><th>Dev. Matthews</th></tr><tr><td>0.596</td></tr></table>|<table><tr><th>Dev. Matthews</th></tr><tr><td>0.600</td></tr></table>|<table><tr><th>Dev. Matthews</th></tr><tr><td>0.432</td></tr></table>|

### STSB

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```
python -m models.bert_glue --model STSB --checkpoint bert-base-uncased --lr 1.86524e-5 --num-labels 1 --max-seq-length 128 --batch-size 16 --seed 637
python -m models.ap_glue --model AP_STSB --checkpoint bert-base-uncased --lr 2.7762e-5 --num-labels 1 --max-seq-length 128 --batch-size 16 --seed 260

python -m models.bert_glue --model STSB --checkpoint bert-large-uncased --lr 7.1099e-6 --num-labels 1 --max-seq-length 128 --batch-size 16 --seed 701
python -m models.ap_glue --model AP_STSB --checkpoint bert-large-uncased --lr 2.98363e-5 --num-labels 1 --max-seq-length 128 --batch-size 16 --seed 131
```

|BERT-base STSB | BERT-large STSB | BERTVision-base STSB |  BERTVision-large STSB |
|--|--|--|--|
|<table><tr><th>Dev. Pearson</th><th>Dev. Spearman</th></tr><tr><td>0.888</td><td>0.886</td></tr></table>|<table><tr><th>Dev. Pearson</th><th>Dev. Spearman</th></tr><tr><td>0.894</td><td>0.891</td></tr></table>|<table><tr><th>Dev. Pearson</th><th>Dev. Spearman</th></tr><tr><td>0.862</td><td>0.859</td></tr></table>|<table><tr><th>Dev. Pearson</th><th>Dev. Spearman</th></tr><tr><td>0.880</td><td>0.879</td></tr></table>|


# Embeddings Replication

## GLUE

To replicate our AdapterPooler models, you must first output 1-epoch fine-tuned
BERT embeddings. The commands below will fine tune BERT, generate the embeddings,
and should be run from the following directory `BERTVision\code\torch\gen_embeds`:

### QQP

```
python qqpairs_embeds.py --checkpoint bert-base-uncased --lr 1e-5 --num-labels 2 --max-seq-length 128 --batch-size 32
python qqpairs_embeds.py --checkpoint bert-large-uncased --lr 1e-5 --num-labels 2 --max-seq-length 128 --batch-size 32
```

### QNLI

```
python qnli_embeds.py --model QNLI --checkpoint bert-base-uncased --lr 1e-5 --num-labels 2 --max-seq-length 128 --batch-size 32
python qnli_embeds.py --model QNLI --checkpoint bert-large-uncased --lr 1e-5 --num-labels 2 --max-seq-length 128 --batch-size 32
```

### MNLI

```
python mnli_embeds.py --checkpoint bert-base-uncased --lr 1e-5 --num-labels 3 --max-seq-length 128 --batch-size 32
python mnli_embeds.py --checkpoint bert-large-uncased --lr 1e-5 --num-labels 3 --max-seq-length 128 --batch-size 32
```

### STSB

```
python stsb_embeds.py --checkpoint bert-base-uncased --lr 2e-5 --num-labels 1 --max-seq-length 128 --batch-size 16
python stsb_embeds.py --checkpoint bert-large-uncased --lr 2e-5 --num-labels 1 --max-seq-length 128 --batch-size 16
```

### CoLA

```
python cola_embeds.py --checkpoint bert-base-uncased --lr 2e-5 --num-labels 2 --max-seq-length 128 --batch-size 16
python cola_embeds.py --checkpoint bert-large-uncased --lr 2e-5 --num-labels 2 --max-seq-length 128 --batch-size 16
```

### MSR

```
python msr_embeds.py --checkpoint bert-base-uncased --lr 2e-5 --num-labels 2 --max-seq-length 128 --batch-size 32
python msr_embeds.py --checkpoint bert-large-uncased --lr 2e-5 --num-labels 2 --max-seq-length 128 --batch-size 32
```

### SST

```
python sst_embeds.py --checkpoint bert-base-uncased --lr 2e-5 --num-labels 2 --max-seq-length 128 --batch-size 32
python sst_embeds.py --checkpoint bert-large-uncased --lr 2e-5 --num-labels 2 --max-seq-length 128 --batch-size 32
```

### RTE

```
python rte_embeds.py --checkpoint bert-base-uncased --lr 2e-5 --num-labels 2 --max-seq-length 250 --batch-size 16
python rte_embeds.py --checkpoint bert-large-uncased --lr 2e-5 --num-labels 2 --max-seq-length 250 --batch-size 16
```


# OLD: TO BE REPLACED

## SQuAD 2.0

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

`python -m models.bert --dataset SQuAD --model bert-base-uncased --num-workers 4 --batch-size 16 --epochs 1 --lr 2e-5 --l2 1.0`

`python -m models.adapter_pooler --dataset SQuADH5 --model AP --num-workers 0 --batch-size 16 --epochs 1 --lr 2e-5 --l2 1.0`

|BERT-base SQuAD 2.0 | BERTVision SQuAD 2.0 |
|--|--|
|<table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>70.3950</td><td>73.5724</td></tr> </table>| <table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>70.0581</td><td>73.3558</td></tr> </table>|

## GLUE

### SST-2

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

`python -m models.sst --dataset SST --model SST --num-workers 0 --batch-size 16 --epochs 1 --lr 1e-5`

`python -m models.ap_sst --dataset SSTH5 --model ap_sst --num-workers 0 --batch-size 16 --epochs 1 --lr 1e-5`

|BERT-base SST-2 | BERTVision SST-2 |
|--|--|
|<table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.91628</td><td>0.91628</td></tr> </table>| <table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.9128</td><td>0.9128</td></tr> </table>|

### MSR
To replicate our results, please run the follow commands from `BERTVision\code\torch`:

`python -m models.msr --dataset MSR --model msr --num-workers 0 --batch-size 16 --epochs 1 --lr 3e-5 --l2 1.0`

`python -m models.ap_msr --dataset MSRH5 --model ap_msr --num-workers 0 --batch-size 16 --epochs 1 --lr 3e-5 --l2 1.0`

|BERT-base MSR | BERTVision MSR |
|--|--|
|<table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.75188</td><td>NA</td></tr> </table>| <table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.7820</td><td>0.7820</td></tr> </table>|

### RTE

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

`python -m models.rte --dataset RTE --model RTE --num-workers 0 --batch-size 16 --epochs 1 --lr 1e-5`

`python -m models.ap_rte --dataset RTEH5 --model ap_rte --num-workers 0 --batch-size 16 --epochs 1 --lr 1e-5`

|BERT-base RTE | BERTVision RTE |
|--|--|
|<table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.5848</td><td>0.5848</td></tr> </table>| <table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.6354</td><td>0.6354</td></tr> </table>|


### QNLI

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

`python -m models.qnli --dataset qnli --model qnli --num-workers 0 --batch-size 16 --epochs 1 --lr 1e-5`

`python -m models.ap_qnli --dataset QNLIH5 --model ap_qnli --num-workers 0 --batch-size 16 --epochs 1 --lr 1e-5`

|BERT-base RTE | BERTVision RTE |
|--|--|
|<table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.9078</td><td>NA</td></tr> </table>| <table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.8948</td><td>0.8948</td></tr> </table>|


### QQPairs

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

`python -m models.qqpairs --dataset QQPairs --model QQPairs --num-workers 0 --batch-size 16 --epochs 1 --lr 1e-5`

`python -m models.ap_qqpairs --dataset QQPairs --model ap_qqpairs --num-workers 0 --batch-size 16 --epochs 1 --lr 1e-5`

|BERT-base RTE | BERTVision RTE |
|--|--|
|<table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.8962</td><td>NA</td></tr> </table>| <table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.8963</td><td>0.8963</td></tr> </table>|


### STSB

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

`python -m models.stsb --dataset STSB --model STSB --num-workers 0 --batch-size 16 --epochs 1 --lr 1e-5`

`python -m models.ap_stsb --dataset STSBH5 --model ap_stsb --num-workers 0 --batch-size 16 --epochs 1 --lr 1e-5`

|BERT-base STSB | BERTVision STSB |
|--|--|
|<table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Pearson.</th><th>Spearman</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.84533</td><td>0.84609</td></tr> </table>| <table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Pearson</th><th>Spearman</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.8571</td><td>0.8568</td></tr> </table>|

### COLA

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

`python -m models.cola --dataset COLA --model COLA --num-workers 0 --batch-size 16 --epochs 1 --lr 1e-5`

`python -m models.ap_cola --dataset COLAH5 --model ap_cola --num-workers 0 --batch-size 16 --epochs 1 --lr 1e-5`

|BERT-base COLA | BERTVision COLA |
|--|--|
|<table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Matthews</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.5131</td></tr> </table>| <table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Matthews.</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.5408</td></tr> </table>|


### MNLI

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

`python -m models.mnli --dataset MNLI --model MNLI --num-workers 0 --batch-size 16 --epochs 1 --lr 1e-5 --l2 1.0`

`python -m models.ap_mnli --dataset MNLIH5 --model ap_mnli --num-workers 0 --batch-size 16 --epochs 1 --lr 1e-5 --l2 1.0`

|BERT-base MNLI | BERTVision MNLI |
|--|--|
|<table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Matched</th><th>Dev. Mismatched</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.8304</td><td>0.8348</td></tr> </table>| <table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Matched</th><th>Dev. Mismatched</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.8287</td><td>0.8303</td></tr> </table>|
