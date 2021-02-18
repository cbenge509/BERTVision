# Create BERTVision Environment

```
conda create -n my_ml python=3.7 pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
conda activate my_ml
conda install -c conda-forge jupyter numpy pandas matplotlib scikit-learn pip python-language-server[all] tensorflow-gpu transformers datasets elasticsearch tqdm
conda install -c install tokenizers
pip install loguru hyperopt
```


# Hyperparameters

We use the following hyperparameters drawing from our own `hyperopt` hyperparameter searches and academic testing.
We find that smaller data sets are much more sensitive to the learning rate than larger ones.

Our hyperparameter search can be run with the following command:

```
python -m models.hypersearch --model MSR --checkpoint bert-large-uncased --batch-size 16 --num-labels 2 --max-seq-length 86
python -m models.ap_hypersearch --model AP_STSB --checkpoint bert-base-uncased --batch-size 16 --num-labels 1 --max-seq-length 77
```

Ensure that you specify the GLUE task `model` as well as the appropriate values for `batch-size`, `num-labels` and `max-seq-length`.

BERT-Large | MNLI | QNLI | QQP | RTE | SST | MSR | CoLA | STS-B
---|---|---|---|---|---|---|---|---
`--num-labels` | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 1
`--lr` | 1e-5 | 1e-5 | 1e-5 | 2e-5 | 2e-5 | 2e-5 | 2e-5 | 2e-5
`--batch-size` | 32 | 32 | 32 | 16 | 32 | 16 | 16 | 16
`--max-seq-length` | 444 | 512 | 330 | 289 | 128 | 128 | 128 | 128

BERT-Base | MNLI | QNLI | QQP | RTE | SST | MSR | CoLA | STS-B
---|---|---|---|---|---|---|---|---
`--num-labels` | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 1
`--lr` | 1e-5 | 1e-5 | 1e-5 | 2e-5 | 2e-5| 2e-5 | 2e-5 | 2e-5
`--batch-size` | 32 | 32 | 32 | 16 | 32 | 16 | 16 | 16
`--max-seq-length` | 444 | 512 | 330 | 289 | 128 | 128 | 128 | 128

# Our Results
BERT-base and AP Models

<img align="center" src="../../images/bert_base_vs_bertvision.png" />


# Latest Results

## GLUE

### QQP

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```
TODO
```

|BERT-base QQP | BERT-large QQP | BERTVision-base QQP |  BERTVision-large QQP |
|--|--|--|--|
|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.887</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.897</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.885</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.896</td></tr></table>|

### QNLI

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```
TODO
```

|BERT-base QNLI | BERT-large QNLI | BERTVision-base QNLI |  BERTVision-large QNLI |
|--|--|--|--|
|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.904</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.915</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.888</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.911</td></tr></table>|

### MNLI

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```
TODO
```

|BERT-base MNLI | BERT-large MNLI | BERTVision-base MNLI |  BERTVision-large MNLI |
|--|--|--|--|
|<table><tr><th>Dev. Matched Acc.</th><th>Dev. Mismatched Acc.</th></tr><tr><td>0.826</td><td>0.831</td></tr></table>|<table><tr><th>Dev. Matched Acc.</th><th>Dev. Mismatched Acc.</th></tr><tr><td>0.857</td><td>0.853</td></tr></table>|<table><tr><th>Dev. Matched Acc.</th><th>Dev. Mismatched Acc.</th></tr><tr><td>0.824</td><td>0.829</td></tr></table>|<table><tr><th>Dev. Matched Acc.</th><th>Dev. Mismatched Acc.</th></tr><tr><td>0.850</td><td>0.847</td></tr></table>|



### RTE

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```
TODO
```

|BERT-base RTE | BERT-large RTE | BERTVision-base RTE |  BERTVision-large RTE |
|--|--|--|--|
|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.661</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.625</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>TODO</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>TODO</td></tr></table>|


### SST

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```
TODO
```

|BERT-base SST | BERT-large SST | BERTVision-base SST |  BERTVision-large SST |
|--|--|--|--|
|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.924</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.929</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>TODO</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>TODO</td></tr></table>|

### MSR

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```
python -m models.bert_glue --model MSR --checkpoint bert-base-uncased --lr 2e-5 --num-labels 2 --max-seq-length 128 --batch-size 32
python -m models.ap_glue --model AP_MSR --checkpoint bert-base-uncased --lr 2e-5 --num-labels 2 --max-seq-length 128 --batch-size 32

python -m models.bert_glue --model MSR --checkpoint bert-large-uncased --lr 2e-5 --num-labels 2 --max-seq-length 128 --batch-size 32
python -m models.ap_glue --model AP_MSR --checkpoint bert-large-uncased --lr 2e-5 --num-labels 2 --max-seq-length 128 --batch-size 32
```

|BERT-base MSR | BERT-large MSR | BERTVision-base MSR |  BERTVision-large MSR |
|--|--|--|--|
|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.810</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.793</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.665</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.829</td></tr></table>|

### CoLA

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```
python -m models.bert_glue --model CoLA --checkpoint bert-base-uncased --lr 2e-5 --num-labels 2 --max-seq-length 128 --batch-size 16
python -m models.ap_glue --model AP_CoLA --checkpoint bert-base-uncased --lr 2e-5 --num-labels 2 --max-seq-length 128 --batch-size 16

python -m models.bert_glue --model CoLA --checkpoint bert-large-uncased --lr 2e-5 --num-labels 2 --max-seq-length 128 --batch-size 16
python -m models.ap_glue --model AP_CoLA --checkpoint bert-large-uncased --lr 2e-5 --num-labels 2 --max-seq-length 128 --batch-size 16
```

|BERT-base CoLA | BERT-large CoLA | BERTVision-base CoLA |  BERTVision-large CoLA |
|--|--|--|--|
|<table><tr><th>Dev. Matthews</th></tr><tr><td>0.464</td></tr></table>|<table><tr><th>Dev. Matthews</th></tr><tr><td>0.567</td></tr></table>|<table><tr><th>Dev. Matthews</th></tr><tr><td>0.518</td></tr></table>|<table><tr><th>Dev. Matthews</th></tr><tr><td>0.409</td></tr></table>|

### STSB

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```
python -m models.bert_glue --model STSB --checkpoint bert-base-uncased --lr 2e-5 --num-labels 1 --max-seq-length 128 --batch-size 16
python -m models.ap_glue --model AP_STSB --checkpoint bert-base-uncased --lr 2e-5 --num-labels 1 --max-seq-length 128 --batch-size 16

python -m models.bert_glue --model STSB --checkpoint bert-large-uncased --lr 2e-5 --num-labels 1 --max-seq-length 128 --batch-size 16
python -m models.ap_glue --model AP_STSB --checkpoint bert-large-uncased --lr 2e-5 --num-labels 1 --max-seq-length 128 --batch-size 16
```

|BERT-base STSB | BERT-large STSB | BERTVision-base STSB |  BERTVision-large STSB |
|--|--|--|--|
|<table><tr><th>Dev. Pearson</th><th>Dev. Spearman</th></tr><tr><td>0.862</td><td>0.856</td></tr></table>|<table><tr><th>Dev. Pearson</th><th>Dev. Spearman</th></tr><tr><td>0.859</td><td>0.856</td></tr></table>|<table><tr><th>Dev. Pearson</th><th>Dev. Spearman</th></tr><tr><td>0.864</td><td>0.865</td></tr></table>|<table><tr><th>Dev. Pearson</th><th>Dev. Spearman</th></tr><tr><td>0.877</td><td>0.875</td></tr></table>|


# Embeddings Replication

## GLUE

To replicate our AdapterPooler models, you must first output 1-epoch fine-tuned
BERT embeddings. The commands below will fine tune BERT, generate the embeddings,
and should be run from the following directory `BERTVision\code\torch\gen_embeds`:

### QQP

```
python qqpairs_embeds.py --checkpoint bert-base-uncased --lr 1e-5 --num-labels 2 --max-seq-length 330 --batch-size 32
python qqpairs_embeds.py --checkpoint bert-large-uncased --lr 1e-5 --num-labels 2 --max-seq-length 330 --batch-size 32
```

### QNLI

```
python qnli_embeds.py --model QNLI --checkpoint bert-base-uncased --lr 1e-5 --num-labels 2 --max-seq-length 512 --batch-size 32
python qnli_embeds.py --model QNLI --checkpoint bert-large-uncased --lr 1e-5 --num-labels 2 --max-seq-length 512 --batch-size 32
```

### MNLI

```
python mnli_embeds.py --checkpoint bert-base-uncased --lr 1e-5 --num-labels 3 --max-seq-length 444 --batch-size 32
python mnli_embeds.py --checkpoint bert-large-uncased --lr 1e-5 --num-labels 3 --max-seq-length 444 --batch-size 32
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
python rte_embeds.py --checkpoint bert-base-uncased --lr 2e-5 --num-labels 2 --max-seq-length 289 --batch-size 16
python rte_embeds.py --checkpoint bert-large-uncased --lr 2e-5 --num-labels 2 --max-seq-length 289 --batch-size 16
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
