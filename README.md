BERT Vision
==========================================================

![GitHub](https://img.shields.io/github/license/cbenge509/BERTVision) ![GitHub Pipenv locked Python version](https://img.shields.io/github/pipenv/locked/python-version/cbenge509/BERTVision) ![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/cbenge509/BERTVision/tensorflow) ![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/cbenge509/BERTVision/transformers) ![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/cbenge509/BERTVision/altair) ![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/cbenge509/BERTVision/h5py)

<img align="right" width="180" src="./images/ucb.png"/>

#### Authors : [William Casey King, PhD](https://jackson.yale.edu/person/casey-king/) | [Cristopher Benge](https://cbenge509.github.io/) | [Siduo (Stone) Jiang](mailto:siduojiang@ischool.berkeley.edu)

[![](https://sourcerer.io/fame/cbenge509/cbenge509/BERTVision/images/0)](https://sourcerer.io/fame/cbenge509/cbenge509/BERTVision/links/0)[![](https://sourcerer.io/fame/cbenge509/cbenge509/BERTVision/images/1)](https://sourcerer.io/fame/cbenge509/cbenge509/BERTVision/links/1)[![](https://sourcerer.io/fame/cbenge509/cbenge509/BERTVision/images/2)](https://sourcerer.io/fame/cbenge509/cbenge509/BERTVision/links/2)[![](https://sourcerer.io/fame/cbenge509/cbenge509/BERTVision/images/3)](https://sourcerer.io/fame/cbenge509/cbenge509/BERTVision/links/3)[![](https://sourcerer.io/fame/cbenge509/cbenge509/BERTVision/images/4)](https://sourcerer.io/fame/cbenge509/cbenge509/BERTVision/links/4)[![](https://sourcerer.io/fame/cbenge509/cbenge509/BERTVision/images/5)](https://sourcerer.io/fame/cbenge509/cbenge509/BERTVision/links/5)[![](https://sourcerer.io/fame/cbenge509/cbenge509/BERTVision/images/6)](https://sourcerer.io/fame/cbenge509/cbenge509/BERTVision/links/6)[![](https://sourcerer.io/fame/cbenge509/cbenge509/BERTVision/images/7)](https://sourcerer.io/fame/cbenge509/cbenge509/BERTVision/links/7)


U.C. Berkeley, Masters in Information & Data Science program - [datascience@berkeley](https://datascience.berkeley.edu/) <br>
Summer 2020, W266 - Natural Language Processing - [Daniel Cer, PhD](https://scholar.google.com/citations?user=BrT1NW8AAAAJ&hl=en) - Section 1

---

<center>
<img src="/images/bertvision.png" height=550 align="center"></center>

## Description

This repository contains the code, models, and documentation for the evaluation of leveraging parameter-efficient models, like those typically used in computer vision, for their potential utility in performing the NLP tasks of span annotation (aka "Q&A") and document binary classification. 

All models were trained on the hidden embedding activation states of $BERT_{LARGE}$ and evaluated on the [Stanford Question Answering Dataset 2.0](https://rajpurkar.github.io/SQuAD-explorer/) (aka [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)).  Our aim for this project is to test the hypothesis that significantly smaller, parameter-efficient models can perform as well or better than BERT on NLP tasks by benefiting from the full 24 layers of transformer block learnings after BERT has been only lightly fine-tuned.

#### Highlight of key files included in this repository:

  |File | Description |
  |:----|:------------|
  | [BERT Vision - Final Paper](/paper/jiang_king_benge__w266_summer_2020_final.pdf) | Our final write-up and submission for review of analysis and findings. |
  | [Generate Binary Classification Embeddings](/notebooks/Generate%20Binary%20Classification%20Embeddings.ipynb) | Jupyter Notebook that walks through the extraction of hiden state activations from BERT (for binary classification) |
  | [Performance Reporting](/notebooks/Performance%20Reporting.ipynb) | Jupyter Notebook used to generate performance tables and plots resulting from model experiments. |
  | [BERT Fine Tuning Binary Classification](/notebooks/BERT%20/Fine%20Tuning%20Binary%20Classification.ipynb) | Jupyter Notebook that walks through fine-tuning BERT for the binary classification task. |
  |[generate_squad_features.py](generate_squad_features.py) | Utility script that generates the train and dev tokens for BERT from the SQuAD v2 dataset |
  
---

Data Pipeline
-------------

For the task of span annotation, BERT was fine-tuned to 6 epochs on the SQuAD 2.0 train dataset and evaluated for accuracy and F1 score at each epoch.  Additionally, fractional epochs 1/10th through 9/10th were captured on the 0th to 1st epoch.  BERT was configured with a sequence length of 386 for all experiments, and during embedding extraction the full sequence length was retained for span annotation:

<img src="/images/Data_Pipeline_Span_Annotation.png" align="center" width = 700>
<br>

For the task of binary classification, BERT was fine-tuned to 6 epochs on the SQuAD 2.0 train dataset and evaluated for accuracy and F1 score at each epoch.  Additionally, fractional epochs 1/10th through 9/10th were captured on the 0th to 1st epoch.  BERT was configured with a sequence length of 386 for all experiments, however the sequence wasn't critical due to BERT's CLS token network.  Here, we retain a 1 x 1024 shape for each example:

<img src="/images/Data_Pipeline_Binary_Classification.png" align="center" width = 700>

---

Results
-------------

This section is still under development, pardon our dust.

<img src="/images/BinaryClassification_BERT_Training_Performance_plot.png" align="center" width = 700>
<br>

<img src="/images/Detail_Tenney_Small_Performance.png" align="center" width = 700>
<br>

<img src="/images/BinaryClassification_Tenney_Small_1_epoch_BERT_fine_tuned_Performance_plot.png" align="center" width = 700>
<br>
<img src="/images/BinaryClassification_Tenney_Small_1_epoch_BERT_fine_tuned_Performance_table.png">
<br>

---


License
-------
Licensed under the MIT License. See [LICENSE](LICENSE.txt) file for more details.
