<img src="./images/bertvision.png" />

# Introduction to BERTVision

We present a highly parameter-efficient approach for a wide range of NLP tasks that significantly reduces the need for extended BERT fine-tuning. Our method uses information from the hidden state activations of each BERT transformer layer, which is discarded during typical BERT inference. Our best model achieves maximal BERT performance at a fraction of the training time and GPU/TPU expense. Performance is further improved by ensembling our model with BERT’s predictions. Furthermore, we find that near optimal performance can be achieved for some NLP tasks using less training data. 

All models were trained on the hidden embedding activation states of BERT-base and BERT-large uncased and evaluated on the [Stanford Question Answering Dataset 2.0](https://rajpurkar.github.io/SQuAD-explorer/) (aka [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)) and the [General Language Understanding Evaluation](https://gluebenchmark.com/) ([GLUE](https://gluebenchmark.com/)) benchmark data.

---

### How it Works: Data Pipeline

Training data for BERTVision is provided by extraction of the embeddings modestly updated within BERT during partial fine-tuning (i.e. fine-tuning using a small fraction of the data).  After partial fine-tuning, the entire training set is inferenced through BERT and embedding values are collected for each sample in the shape of (386,1024,25) for SQuAD (and similar for GLUE) - much like an image represetened in the shape of (H,W,C).  These training "images" are then fit to a much smaller model using our AdapterPooler technique, and are reduced along the depth dimension using a simple linear pooling technique adapted from [Tenney et. al](https://arxiv.org/pdf/1905.05950.pdf)'s *edge probing* method.

**Depicted below: extraction of data for Span Annotation Task**

<img src="./images/Data_Pipeline_Span_Annotation.png" />

---

### How it Works: Model Architecture

BERT embeddings from all encoder layers are first transformed through our customer adapter layer (referred to as *LayerWeightShare* in the paper).  Next, the last two dimensions output from the adapter are flattened, and a residual skip connection to the original input is combined with them before being projected down for final inferencing.  Depicted below is the architecture for the span annotation task; the tensor is projected down to a size of (386,2) with a densely connected layer and split on the last axis into two model heads.  These represent the logits of the start-span and end-span position for the span annotation task; for other tasks, the output sequence varies depending on the task goal.

<center><img src="./images/BERTVision_QA_Model.png" width=400 align="center" /></center>
<br>

---

### How it Works: Model Development & Training

Our development and experiementation was performed in an Infrastructure-as-a-Service topology consisting of two NVIDIA Tesla V100 GPU-backed virtual machines in the Microsoft Azure cloud.  Data was stored on virtually attached SSD's utilizing approximately 20TiB combined.  Our development enviornment consisted of Python v3.8.5, TensorFLow v2.4.1, and PyTorch v1.7.1. Visualization support was provided primarily through the Altair v4.1.0 and Plotly v4.14.3 libraries, and all documentation was managed through LaTeX.  Azure DevOps (Boards) and GitHub (repositories) were used to managed project and code, respectively.

<img src="./images/BERTVision_development_pipeline.png" />

---

# NLP Tasks & Data

We evaluated the effectiveness and efficiency of BERTVision on two industry benchmark datasets:  [The General Language Understanding Evaluation (GLUE)](https://gluebenchmark.com/) benchmark, and the [Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/) $v2.0$.
### SQuAD 2.0 Datasets

The Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable. SQuAD v2.0 combines the 100,000 questions in SQuAD v1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. To do well on SQuAD2.0, systems must not only answer questions when possible, but also determine when no answer is supported by the paragraph and abstain from answering.

| Dataset | Description | NLP Task | Metric | Size |
|:--------|:------------|:--------:|:------:|------|
| [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) | A reading comprehension dataset prepared by crowdworkers on a set of Wikipedia articles. | [span annotation](https://en.wikipedia.org/wiki/Question_answering), [classification](https://en.wikipedia.org/wiki/Binary_classification) | [Exact Match](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2761899.pdf), [F1](https://en.wikipedia.org/wiki/F-score) | ~150k |

### GLUE Benchmark Datasets

The General Language Understanding Evaluation (GLUE) benchmark is a collection of resources for training, evaluating, and analyzing natural language understanding systems. GLUE consists of:

 - A benchmark of nine sentence- or sentence-pair language understanding tasks built on established existing datasets and selected to cover a diverse range of dataset sizes, text genres, and degrees of difficulty,
 - A diagnostic dataset designed to evaluate and analyze model performance with respect to a wide range of linguistic phenomena found in natural language, and
 - A public leaderboard for tracking performance on the benchmark and a dashboard for visualizing the performance of models on the diagnostic set.

The format of the GLUE benchmark is model-agnostic, so any system capable of processing sentence and sentence pairs and producing corresponding predictions is eligible to participate. The benchmark tasks are selected so as to favor models that share information across tasks using parameter sharing or other transfer learning techniques. The ultimate goal of GLUE is to drive research in the development of general and robust natural language understanding systems.

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
| [WNLI](https://cs.nyu.edu/faculty/davise/papers/WinogradSchemas/WS.html) | Winnograd Natural Lanugage Inference | Natural Language Inference | Accuracy | ~1k |

---

# Our Results

### Hyperparameter Searching

We use `hyperopt` to search over parameters and tune our models. We find that the smaller data sets are far more sensitive to tuning than the larger ones. To replicate our tuning processes, please use the following commands:

```bash
python -m models.hypersearch --model MSR --checkpoint bert-large-uncased --batch-size 32 --num-labels 2 --max-seq-length 128
python -m models.ap_hypersearch --model AP_STSB --checkpoint bert-base-uncased --batch-size 16 --num-labels 1 --max-seq-length 128
```

For large data sets, e.g., MNLI, QNLI, QQP, and SST, data set sharding is enabled automatically, which randomly samples 10% of the data set to train on to speed up the parameter search. shard is manipulable and can be set by:

```bash
python -m models.ap_hypersearch --model AP_QQP --checkpoint bert-base-uncased --batch-size 32 --num-labels 2 --max-seq-length 128 --shard 0.
```

The table below displays the commonly recommended general hyperparameters for each GLUE task. The BERTVision embeddings were generated based on these parameters:

| BERT-(base/large) | MNLI | QNLI | QQP | RTE | SST | MSR | CoLA | STS-B |
|:------------------|:-----|:-----|:----|:----|:----|:----|:-----|:------|
| `--num-labels`    | 3    | 2    | 2   | 2   | 2   | 2   | 2    | 1     |
| `--lr`            | 1e-5 | 1e-5 | 1e-5|2e-5 | 2e-5|2e-5 | 2e-5 | 2e-5  |
| `--batch-size`    | 32   | 32   | 32  | 16  | 32  | 32  | 16   | 16    |
| `--max-seq-length`| 128  | 128  | 128 | 250 | 128 | 128 | 128  | 128   |

### Results Table : BERT-(base/large} vs. BERTVision (All Tasks)

<img src="./images/bert_base_vs_bertvision.png"/>



### Results : SQuAD 2.0

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```bash
python -m models.bert --dataset SQuAD --model bert-base-uncased --num-workers 4 
    --batch-size 16 --epochs 1 --lr 2e-5 --l2 1.0
python -m models.adapter_pooler --dataset SQuADH5 --model AP --num-workers 0 
    --batch-size 16 --epochs 1 --lr 2e-5 --l2 1.0`
```

<table>
    <tr>
        <th>Model</th><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th>
    </tr>
    <tr>
        <td>BERT-base</td><td>1</td><td>16</td><td>1.0</td><td>70.3950</td><td>73.5724</td>
    </tr> 
    <tr>
        <td>BERTVision</td><td>1</td><td>16</td><td>1.0</td><td>70.0581</td><td>73.3558</td>
    </tr>
</table>

### Results (GLUE) : SST-2

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```bash
python -m models.sst --dataset SST --model SST --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
python -m models.ap_sst --dataset SSTH5 --model ap_sst --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5`
```

<table>
    <tr>
        <th>Model</th><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th>
    </tr>
    <tr>
        <td>BERT-base</td><td>1</td><td>16</td><td>1.0</td><td>0.91628</td><td>0.91628</td>
    </tr>
    <tr>
        <td>BERTVision</td><td>1</td><td>16</td><td>1.0</td><td>0.9128</td><td>0.9128</td>
    </tr>
</table>

### Results (GLUE) : MSR
To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```bash
python -m models.msr --dataset MSR --model msr --num-workers 0 
    --batch-size 16 --epochs 1 --lr 3e-5 --l2 1.0
python -m models.ap_msr --dataset MSRH5 --model ap_msr --num-workers 0 
    --batch-size 16 --epochs 1 --lr 3e-5 --l2 1.0
```

<table>
    <tr>
        <th>Model</th><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th>
    </tr>
    <tr>
        <td>BERT-base</td><td>1</td><td>16</td><td>1.0</td><td>0.75188</td><td>NA</td>
    </tr>
    <tr>
        <td>BERTVision</td><td>1</td><td>16</td><td>1.0</td><td>0.7820</td><td>0.7820</td>
    </tr>
</table>

### Results (GLUE) : RTE

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```bash
python -m models.rte --dataset RTE --model RTE --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
python -m models.ap_rte --dataset RTEH5 --model ap_rte --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
```

<table>
    <tr>
        <th>Model</th><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th>
    </tr>
    <tr>
        <td>BERT-base</td><td>1</td><td>16</td><td>1.0</td><td>0.5848</td><td>0.5848</td>
    </tr>
    <tr>
        <td>BERTVision</td><td>1</td><td>16</td><td>1.0</td><td>0.6354</td><td>0.6354</td>
    </tr>
</table>

### Results (GLUE) : QNLI

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```bash
python -m models.qnli --dataset qnli --model qnli --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
python -m models.ap_qnli --dataset QNLIH5 --model ap_qnli --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
```

<table>
    <tr>
        <th>Model</th><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th>
    </tr>
    <tr>
        <td>BERT-base</td><td>1</td><td>16</td><td>1.0</td><td>0.9078</td><td>NA</td>
    </tr>
    <tr>
        <td>BERTVision</td><td>1</td><td>16</td><td>1.0</td><td>0.8948</td><td>0.8948</td>
    </tr>
</table>

### Results (GLUE) : QQPairs

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```bash
python -m models.qqpairs --dataset QQPairs --model QQPairs --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
python -m models.ap_qqpairs --dataset QQPairs --model ap_qqpairs --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
```

<table>
    <tr>
        <th>Model</th><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Acc.</th><th>Dev. F1</th>
    </tr>
    <tr>
        <td>BERT-base</td><td>1</td><td>16</td><td>1.0</td><td>0.8962</td><td>NA</td>
    </tr>
    <tr>
        <td>BERTVision</td><td>1</td><td>16</td><td>1.0</td><td>0.8963</td><td>0.8963</td>
    </tr>
</table>

### Results (GLUE) : STSB

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```bash
python -m models.stsb --dataset STSB --model STSB --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
python -m models.ap_stsb --dataset STSBH5 --model ap_stsb --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
```

<table>
    <tr>
        <th>Model</th><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Pearson.</th><th>Spearman</th>
    </tr>
    <tr>
        <td>BERT-base</td><td>1</td><td>16</td><td>1.0</td><td>0.84533</td><td>0.84609</td>
    </tr>
    <tr>
        <td>BERTVision</td><td>1</td><td>16</td><td>1.0</td><td>0.8571</td><td>0.8568</td>
    </tr>
</table>

### Results (GLUE) : COLA

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```bash
python -m models.cola --dataset COLA --model COLA --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
python -m models.ap_cola --dataset COLAH5 --model ap_cola --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5
```

<table>
    <tr>
        <th>Model</th><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Matthews</th>
    </tr>
    <tr>
        <td>BERT-base</td><td>1</td><td>16</td><td>1.0</td><td>0.5131</td>
    </tr>
    <tr>
        <td>BERTVision</td><td>1</td><td>16</td><td>1.0</td><td>0.5408</td>
    </tr>
</table>

### Results (GLUE) : MNLI

To replicate our results, please run the follow commands from `BERTVision\code\torch`:

```bash
python -m models.mnli --dataset MNLI --model MNLI --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5 --l2 1.0
python -m models.ap_mnli --dataset MNLIH5 --model ap_mnli --num-workers 0 
    --batch-size 16 --epochs 1 --lr 1e-5 --l2 1.0
```

<table>
    <tr>
        <th>Model</th><th>Epoch</th><th>Batch Size</th><th>L2</th><th>Dev. Matched</th><th>Dev. Mismatched</th>
    </tr>
    <tr>
        <td>BERT-base</td><td>1</td><td>16</td><td>1.0</td><td>0.8304</td><td>0.8348</td>
    </tr>
    <tr>
        <td>BERTVision</td><td>1</td><td>16</td><td>1.0</td><td>0.8287</td><td>0.8303</td>
    </tr>
</table>

---

# Read the Paper

The BERTVision paper has been published, and is yadda yadda yadda... :

**[BERTVision: A Parameter-Efficient Approach for BERT-based NLP Tasks](./paper/BERTVision_2020.pdf)**

