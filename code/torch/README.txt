
Conda environment:
conda create -n my_ml python=3.7 jupyter numpy pandas matplotlib scikit-learn pip 'ipykernel<5.0.0' python-language-server[all] tensorflow-gpu nltk pytorch torchvision cudatoolkit=10.1 huggingface transformers tokenizer datasets elasticsearch tqdm


To run on CLI:
conda activate my_ml
cd c:\BERTVision\code\torch


### BERT-QA on SQuAD 2.0:
python -m models.bert --dataset SQuAD --model bert-base-uncased --num-workers 4 --batch-size 16 --epochs 3

### AdapterPooler on BERT-QA SQuAD 2.0 embeddings
python -m models.adapter_pooler --dataset SQuADH5 --model AP --num-workers 0 --batch-size 16 --epochs 3 --lr 1e-3

### SST-2 on BERT for SequenceClassification
python -m models.sst --dataset SST --model SST --num-workers 0 --batch-size 16 --epochs 3 --lr 1e-5

### SST-5 on BERT for SequenceClassification
python -m models.sst --dataset SST --model SST --num-workers 0 --batch-size 16 --epochs 3 --lr 1e-5 --is-multilabel



# some results

bert qa
epoch  exact,    f1,      loss dev
 1      71.3131  74.6153  1.0209
 2      71.8689  75.5646  1.1189
 3      73.460   76.893   1.237

adapter pooler
exact,    f1,      loss dev
69.013  71.9318    1.05

