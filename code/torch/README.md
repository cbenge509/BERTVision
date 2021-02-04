# Create BERTVision Environment

```
conda create -n my_ml python=3.7 pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
conda activate my_ml
conda install -c conda-forge jupyter numpy pandas matplotlib scikit-learn pip python-language-server[all] tensorflow-gpu transformers datasets elasticsearch tqdm
conda install -c install tokenizers
pip install pytreebank
```

# Our Results


## SQuAD 2.0

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

`python -m models.bert --dataset SQuAD --model bert-base-uncased --num-workers 4 --batch-size 16 --epochs 1 --lr 3e-5 --l2 1.0`

`python -m models.adapter_pooler --dataset SQuADH5 --model AP --num-workers 0 --batch-size 16 --epochs 1 --lr 3e-5 --l2 1.0`

|BERT-base SQuAD 2.0 | BERTVision SQuAD 2.0 |
|--|--|
|<table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Loss</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.9970</td><td>72.0542</td><td>75.0912</td></tr> </table>| <table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Loss</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>Dev. Loss</td><td>Dev. Acc.</td><td>Dev. F1</td></tr> </table>|

## GLUE

### SST-2

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

`python -m models.sst --dataset SST --model SST --num-workers 0 --batch-size 16 --epochs 3 --lr 1e-5`

`python -m models.ap_sst --dataset SSTH5 --model ap_sst --num-workers 0 --batch-size 16 --epochs 3 --lr 1e-5`

|BERT-base SST-2 | BERTVision SST-2 |
|--|--|
|<table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Loss</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.2958</td><td>0.8856</td><td>0.8856</td></tr> </table>| <table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Loss</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.3351</td><td>0.8750</td><td>0.8750</td></tr> </table>|


### MSR

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

`python -m models.msr --dataset MSR --model msr --num-workers 0 --batch-size 16 --epochs 1 --lr 3e-5 --l2 1.0`

`python -m models.ap_msr --dataset MSRH5 --model ap_msr --num-workers 0 --batch-size 16 --epochs 1 --lr 3e-5 --l2 1.0`

|BERT-base MSR | BERTVision MSR |
|--|--|
|<table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Loss</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.6030</td><td>0.75188</td><td>NA</td></tr> </table>| <table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Loss</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.8383</td><td>0.7820</td><td>0.7820</td></tr> </table>|



# OLD TO BE REPLACED

## SQuAD 2.0 Commands
### BERT-QA on SQuAD 2.0:
python -m models.bert --dataset SQuAD --model bert-base-uncased --num-workers 4 --batch-size 16 --epochs 3

### AdapterPooler on BERT-QA SQuAD 2.0 embeddings
python -m models.adapter_pooler --dataset SQuADH5 --model AP --num-workers 0 --batch-size 16 --epochs 3 --lr 1e-3

# Run Models
### BERT-QA on SQuAD 2.0:
python -m models.bert --dataset SQuAD --model bert-base-uncased --num-workers 4 --batch-size 16 --epochs 3

### AdapterPooler on BERT-QA SQuAD 2.0 embeddings
python -m models.adapter_pooler --dataset SQuADH5 --model AP --num-workers 0 --batch-size 16 --epochs 3 --lr 1e-3

### SST-2 on BERT for SequenceClassification
python -m models.sst --dataset SST --model SST --num-workers 0 --batch-size 16 --epochs 3 --lr 1e-5

### SST-5 on BERT for SequenceClassification
python -m models.sst --dataset SST --model SST --num-workers 0 --batch-size 16 --epochs 3 --lr 1e-5 --is-multilabel

### AdapterPooler SST-2 for SequenceClassification
python -m models.ap_sst --dataset SSTH5 --model ap_sst --num-workers 0 --batch-size 16 --epochs 3 --lr 1e-5

### AdapterPooler MSR for SequenceClassification
python -m models.ap_msr --dataset MSRH5 --model ap_msr --num-workers 0 --batch-size 16 --epochs 3 --lr 3e-5

***GLUE Tasks***

RTE -- Accuracy
C:\BERTVision\code\torch>python -m models.rte --dataset SST --model RTE --num-workers 0 --batch-size 16 --epochs 100 --lr 1e-5

| BERT-Base Dev (Us)      | BERT-Large Reported |
| ----------- | ----------- |
| 0.657 (epoch 4)      | 0.701       |
| 0.668 (epoch 16)   |         |

QNLI -- Accuracy
python -m models.qnli --dataset SST --model QNLI --num-workers 0 --batch-size 16 --epochs 30 --lr 1e-5

| BERT-Base Dev (Us)      | BERT-Large Reported |
| ----------- | ----------- |
| 0.895 (epoch 2)      | 0.927       |
| 0.904 (epoch 3)   |         |

QQPairs -- Accuracy

| BERT-Base Dev (Us)      | BERT-Large Reported |
| ----------- | ----------- |
| 0.908 (epoch 2)      | 0.893       |
| 0.911 (epoch 3)   |         |
