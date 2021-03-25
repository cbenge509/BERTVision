# Create BERTVision Environment

We recommend generating the below Anaconda environment to replicate our results:

```
conda create -n my_ml python=3.7 pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
conda activate my_ml
conda install -c conda-forge jupyter numpy pandas matplotlib scikit-learn pip python-language-server[all] tensorflow-gpu transformers datasets elasticsearch tqdm
conda install -c install tokenizers
pip install loguru hyperopt
```


# Hyperparameter Searching

We use `hyperopt` to search over parameters and tune our models. We find that
the smaller data sets are far more sensitive to tuning than the larger ones.
To replicate our tuning processes, please use the following commands:

```
python -m models.hypersearch --model MSR --checkpoint bert-large-uncased --batch-size 32 --num-labels 2 --max-seq-length 128
python -m models.ap_hypersearch --model AP_STSB --checkpoint bert-base-uncased --batch-size 16 --num-labels 1 --max-seq-length 128
```

For large data sets, e.g., MNLI, QNLI, QQP, and SST, data set sharding is enabled
automatically, which randomly samples 10% of the data set to train on to speed
up the parameter search. `shard` is manipulable and can be set by:

```
python -m models.ap_hypersearch --model AP_QQP --checkpoint bert-base-uncased --batch-size 32 --num-labels 2 --max-seq-length 128 --shard 0.3
```

The table below displays the commonly recommended general hyperparameters for
each GLUE task. The BERTVision embeddings were generated based on these parameters.

BERT-(base/large) | MNLI | QNLI | QQP | RTE | SST | MSR | CoLA | STS-B
---|---|---|---|---|---|---|---|---
`--num-labels` | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 1
`--lr` | 1e-5 | 1e-5 | 1e-5 | 2e-5 | 2e-5 | 2e-5 | 2e-5 | 2e-5
`--batch-size` | 32 | 32 | 32 | 16 | 32 | 32 | 16 | 16
`--max-seq-length` | 128 | 128 | 128 | 250 | 128 | 128 | 128 | 128


# Our Results
BERT-base and AP Models

<img align="center" src="../../images/bert_base_vs_bertvision.png" />


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
python -m models.ap_glue --model AP_RTE --checkpoint bert-base-uncased --lr 0.0003593259178474023 --num-labels 2 --max-seq-length 250 --adapter-dim 8 --batch-size 16 --seed 926

python -m models.bert_glue --model RTE --checkpoint bert-large-uncased --lr 8.3621e-6 --num-labels 2 --max-seq-length 250 --batch-size 16 --seed 244
python -m models.ap_glue --model AP_RTE --checkpoint bert-large-uncased --lr 1.2614e-5 --num-labels 2 --max-seq-length 250 --batch-size 16 --seed 414
```

|BERT-base RTE | BERT-large RTE | BERTVision-base RTE |  BERTVision-large RTE |
|--|--|--|--|
|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.657</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.664</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.726</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.592</td></tr></table>|

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
python -m models.ap_glue --model AP_MSR --checkpoint bert-base-uncased --lr 0.0007591458513071305 --num-labels 2 --max-seq-length 128 --adapter-dim 16 --batch-size 16 --seed 926

python -m models.bert_glue --model MSR --checkpoint bert-large-uncased --lr 1.2771e-5 --num-labels 2 --max-seq-length 128 --batch-size 32 --seed 158
python -m models.ap_glue --model AP_MSR --checkpoint bert-large-uncased --lr 1.4399e-5 --num-labels 2 --max-seq-length 128 --batch-size 32 --seed 465
```

|BERT-base MSR | BERT-large MSR | BERTVision-base MSR |  BERTVision-large MSR |
|--|--|--|--|
|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.828</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.768</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.840</td></tr></table>|<table><tr><th>Dev. Accuracy</th></tr><tr><td>0.837</td></tr></table>|

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


## SQuAD 2.0

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```
python -m models.bert_squad --model SQuAD --checkpoint bert-base-uncased --lr 2e-5 --max-seq-length 384 --batch-size 16
python -m models.ap_squad --model AP_SQuAD --checkpoint bert-base-uncased --lr 2e-5 --max-seq-length 384 --batch-size 16

python -m models.bert_squad --model SQuAD --checkpoint bert-large-uncased --lr 2e-5 --max-seq-length 384 --batch-size 8
python -m models.ap_squad --model AP_SQuAD --checkpoint bert-large-uncased --lr 2e-5 --max-seq-length 384 --batch-size 8
```

|BERT-base SQuAD | BERT-large SQuAD | BERTVision-base SQuAD |  BERTVision-large SQuAD |
|--|--|--|--|
|<table><tr><th>Dev. Exact</th><th>Dev. F1</th></tr><tr><td>69.410</td><td>72.571</td></tr></table>|<table><tr><th>Dev. Exact</th><th>Dev. F1</th></tr><tr><td>77.579</td><td>80.583</td></tr></table>|<table><tr><th>Dev. Exact</th><th>Dev. F1</th></tr><tr><td>70.100</td><td>73.416</td></tr></table>|<table><tr><th>Dev. Exact</th><th>Dev. F1</th></tr><tr><td>76.956</td><td>80.274</td></tr></table>|

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


## SQuAD 2.0

```
python squad_embeds.py --checkpoint bert-base-uncased --lr 2e-5 --max-seq-length 384 --batch-size 16
python squad_embeds.py --checkpoint bert-large-uncased --lr 2e-5 --max-seq-length 384 --batch-size 8
```



# Parameter Freezing

```
python -m models.pfreezing --model MSR --checkpoint bert-base-uncased --batch-size 32 --lr 2e-5 --num-labels 2 --max-seq-length 128
```



# Error Analysis

To enable error analysis on MSR or RTE, the following flag is required:

```
python -m models.bert_glue --model MSR --checkpoint bert-base-uncased --lr 2.4380e-5 --num-labels 2 --max-seq-length 128 --batch-size 32 --seed 734 --error
python -m models.ap_glue --model AP_MSR --checkpoint bert-base-uncased --lr 2.7181e-5 --num-labels 2 --max-seq-length 128 --batch-size 32 --seed 760 --error
```
