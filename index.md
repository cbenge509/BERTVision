<img src="./images/bertvision.png" />

# Introduction to BERTVision

We present a highly parameter-efficient approach for a wide range of NLP tasks that significantly reduces the need for extended BERT fine-tuning. Our method uses information from the hidden state activations of each BERT transformer layer, which is discarded during typical BERT inference. Our best model achieves maximal BERT performance at a fraction of the training time and GPU/TPU expense. Performance is further improved by ensembling our model with BERT’s predictions. Furthermore, we find that near optimal performance can be achieved for some NLP tasks using less training data. 

All models were trained on the hidden embedding activation states of BERT-base and BERT-large uncased and evaluated on the [Stanford Question Answering Dataset 2.0](https://rajpurkar.github.io/SQuAD-explorer/) (aka [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)) and the [General Language Understanding Evaluation](https://gluebenchmark.com/) ([GLUE](https://gluebenchmark.com/)) benchmark data.

---

### How it Works: Data Pipeline

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

<img src="./images/Data_Pipeline_Span_Annotation.png" />

---

### How it Works: Model Architecture

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

<center><img src="./images/BERTVision_QA_Model.png" width=400 align="center" /></center>
<br>

---

### How it Works: Model Development & Training

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

<img src="./images/BERTVision_development_pipeline.png" />

---

# NLP Tasks & Data

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

#### SQuAD 2.0 Datasets

| Dataset | Description | NLP Task | Metric | Size |
|:--------|:------------|:--------:|:------:|------|
| [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) | A reading comprehension dataset prepared by crowdworkers on a set of Wikipedia articles. | [span annotation](https://en.wikipedia.org/wiki/Question_answering), [classification](https://en.wikipedia.org/wiki/Binary_classification) | [Exact Match](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2761899.pdf), [F1](https://en.wikipedia.org/wiki/F-score) | ~150k |

#### GLUE Benchmark Datasets

| Dataset | Description | NLP Task | Metric | Size |
|:--------|:------------|:--------:|:------:|------|
| [CoLA](https://nyu-mll.github.io/CoLA/) | Corpus of Linguistic Acceptability | Acceptability | Matthews | ~10k |
| [SST-2](https://nlp.stanford.edu/sentiment/index.html) | Standford Sentiment Treebank | Sentiment | Accuracy | ~67k |
| [MSR](https://www.microsoft.com/en-us/download/details.aspx?id=52398) | Microsoft Research Paraphrase Corpus | Paraphrase | Accuracy, F1 | ~4k |
| [STS-B](https://www.aclweb.org/anthology/S17-2001) | Semantic Textual Similarity Benchmark | Sentence Similarity | Pearson / Spearman | ~7k |
| [QQPairs](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs) | Quora Question Pairs | Paraphrase | Accuracy, F1 | ~400k |
| [MNLI](https://www.nyu.edu/projects/bowman/multinli/) | Multi-Genre Natural Language Inference Corpus | Natural Language Inference | Accuracy | . |
| [QNLI](https://rajpurkar.github.io/SQuAD-explorer/) | SQuAD dataset | QA / NLI | Accuracy | ~110k |
| [RTE](https://aclweb.org/aclwiki/Textual_Entailment_Resource_Pool) | Recognizing Textual Entailment | Natural Language Inference | Accuracy | ~3k |
| [WNLI](https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html) | Natural Language Inference | Accuracy | ~1k |

---

# Our Results

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

<img src="./images/bert_base_vs_bertvision.png"/>

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.


### Results : SQuAD 2.0

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```bash
python -m models.bert --dataset SQuAD --model bert-base-uncased --num-workers 4 
    --batch-size 16 --epochs 1 --lr 2e-5 --l2 1.0
python -m models.adapter_pooler --dataset SQuADH5 --model AP --num-workers 0 
    --batch-size 16 --epochs 1 --lr 2e-5 --l2 1.0`
```

|BERT-base SQuAD 2.0 | BERTVision SQuAD 2.0 |
|--|--|
|<table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>70.3950</td><td>73.5724</td></tr> </table>| <table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>70.0581</td><td>73.3558</td></tr> </table>|

### Results (GLUE) : SST-2

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```bash
python -m models.sst --dataset SST --model SST --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
python -m models.ap_sst --dataset SSTH5 --model ap_sst --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5`
```

|BERT-base SST-2 | BERTVision SST-2 |
|--|--|
|<table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.91628</td><td>0.91628</td></tr> </table>| <table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.9128</td><td>0.9128</td></tr> </table>|

### Results (GLUE) : MSR
To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```bash
python -m models.msr --dataset MSR --model msr --num-workers 0 
    --batch-size 16 --epochs 1 --lr 3e-5 --l2 1.0
python -m models.ap_msr --dataset MSRH5 --model ap_msr --num-workers 0 
    --batch-size 16 --epochs 1 --lr 3e-5 --l2 1.0
```

|BERT-base MSR | BERTVision MSR |
|--|--|
|<table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.75188</td><td>NA</td></tr> </table>| <table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.7820</td><td>0.7820</td></tr> </table>|

### Results (GLUE) : RTE

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```bash
python -m models.rte --dataset RTE --model RTE --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
python -m models.ap_rte --dataset RTEH5 --model ap_rte --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
```

|BERT-base RTE | BERTVision RTE |
|--|--|
|<table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.5848</td><td>0.5848</td></tr> </table>| <table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.6354</td><td>0.6354</td></tr> </table>|


### Results (GLUE) : QNLI

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```bash
python -m models.qnli --dataset qnli --model qnli --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
python -m models.ap_qnli --dataset QNLIH5 --model ap_qnli --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
```

|BERT-base RTE | BERTVision RTE |
|--|--|
|<table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.9078</td><td>NA</td></tr> </table>| <table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.8948</td><td>0.8948</td></tr> </table>|

### Results (GLUE) : QQPairs

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```bash
python -m models.qqpairs --dataset QQPairs --model QQPairs --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
python -m models.ap_qqpairs --dataset QQPairs --model ap_qqpairs --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
```

|BERT-base RTE | BERTVision RTE |
|--|--|
|<table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.8962</td><td>NA</td></tr> </table>| <table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.8963</td><td>0.8963</td></tr> </table>|

### Results (GLUE) : STSB

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```bash
python -m models.stsb --dataset STSB --model STSB --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
python -m models.ap_stsb --dataset STSBH5 --model ap_stsb --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
```

|BERT-base STSB | BERTVision STSB |
|--|--|
|<table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Pearson.</th><th>Spearman</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.84533</td><td>0.84609</td></tr> </table>| <table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Pearson</th><th>Spearman</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.8571</td><td>0.8568</td></tr> </table>|

### Results (GLUE) : COLA

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```bash
python -m models.cola --dataset COLA --model COLA --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
python -m models.ap_cola --dataset COLAH5 --model ap_cola --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
```

|BERT-base COLA | BERTVision COLA |
|--|--|
|<table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Matthews</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.5131</td></tr> </table>| <table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Matthews.</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.5408</td></tr> </table>|


### Results (GLUE) : MNLI

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```bash
python -m models.mnli --dataset MNLI --model MNLI --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5 --l2 1.0
python -m models.ap_mnli --dataset MNLIH5 --model ap_mnli --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5 --l2 1.0
```

|BERT-base MNLI | BERTVision MNLI |
|--|--|
|<table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Matched</th><th>Dev. Mismatched</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.8304</td><td>0.8348</td></tr> </table>| <table> <tr><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Matched</th><th>Dev. Mismatched</th></tr><tr><td>1</td><td>16</td><td>1.0</td><td>0.8287</td><td>0.8303</td></tr> </table>|

---

# Read the Paper

The BERTVision paper has been published, and is yadda yadda yadda... :

**[BERTVision: A Parameter-Efficient Approach for BERT-based NLP Tasks](./paper/BERTVision_2020.pdf)**




```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/cbenge509/BERTVision/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
