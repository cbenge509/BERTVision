# Create BERTVision Environment

```
conda create -n my_ml python=3.7 pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
conda activate my_ml
conda install -c conda-forge jupyter numpy pandas matplotlib scikit-learn pip python-language-server[all] tensorflow-gpu transformers datasets elasticsearch tqdm
conda install -c install tokenizers
pip install loguru hyperopt
```


# New Model Code:

Will clean this up soon

## bert-base
```
python -m models.rte --model RTE --checkpoint bert-base-uncased
python -m models.ap_rte --model AP_RTE --checkpoint bert-base-uncased

python -m models.wnli --model WNLI --checkpoint bert-base-uncased

python -m models.sst --model SST --checkpoint bert-base-uncased

python -m models.msr --model MSR --checkpoint bert-base-uncased

python -m models.mnli --model MNLI --checkpoint bert-base-uncased

python -m models.qqpairs --model QQPairs --checkpoint bert-base-uncased

python -m models.qnli --model QNLI --checkpoint bert-base-uncased

python -m models.cola --model CoLA --checkpoint bert-base-uncased

python -m models.stsb --model STSB --checkpoint bert-base-uncased

python -m models.squad --model SQuAD --checkpoint bert-base-uncased
```
## bert-large
```
python -m models.rte --model RTE --checkpoint bert-large-uncased
python -m models.ap_rte --model AP_RTE --checkpoint bert-large-uncased

python -m models.wnli --model WNLI --checkpoint bert-large-uncased

python -m models.sst --model SST --checkpoint bert-large-uncased

python -m models.msr --model MSR --checkpoint bert-large-uncased

python -m models.mnli --model MNLI --checkpoint bert-large-uncased

python -m models.qqpairs --model QQPairs --checkpoint bert-large-uncased

python -m models.qnli --model QNLI --checkpoint bert-large-uncased

python -m models.cola --model CoLA --checkpoint bert-large-uncased

python -m models.stsb --model STSB --checkpoint bert-large-uncased

python -m models.squad --model SQuAD --checkpoint bert-large-uncased
```

# Hyperparameters

We use the following hyperparameters drawing from our own `hyperopt` hyperparameter searches and academic testing.
We find that smaller data sets are much more sensitive to the learning rate than larger ones.

Our hyperparameter search can be run with the following command:

`python -m models.hypersearch --model MSR --checkpoint bert-large-uncased --batch-size 16 --num-labels 2 --max-seq-length 86`

Ensure that you specify the GLUE task `model` as well as the appropriate values for `batch-size`, `num-labels` and `max-seq-length`.

BERT-Large | MNLI | QNLI | QQP | RTE | SST-2 | MSR | CoLA | STS-B
---|---|---|---|---|---|---|---|---
`--num-labels` | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 1
`--lr` | 1e-5 | 1e-5 | 1e-5 | 1.37018e-05 | 1e-5 | 1.0552e-05 | 1.3829e-05 | 1.18555e-05
`--batch-size` | 32 | 32 | 32 | 16 | 32 | 16 | 16 | 16
`--max-seq-length` | 114 | 121 | 84 | 219 | 66 | 86 | 64 | 77

BERT-Base | MNLI | QNLI | QQP | RTE | SST-2 | MSR | CoLA | STS-B
---|---|---|---|---|---|---|---|---
`--num-labels` | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 1
`--lr` | 1e-5 | 1e-5 | 1e-5 | 2e-5 | 2.87889e-05| 2.0178e-05 | 2.3571e-05 | 2.1123e-05
`--batch-size` | 32 | 32 | 32 | 16 | 32 | 16 | 16 | 16
`--max-seq-length` | 114 | 121 | 84 | 219 | 66 | 86 | 64 | 77

# Our Results
BERT-base and AP Models

<img align="center" src="../../images/bert_base_vs_bertvision.png" />

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
