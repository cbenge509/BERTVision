Only contains BERT for QA and SQuAD 2.0 in the new framework.

To run on CLI:

conda activate my_ml

cd c:\BERTVision

BERT-QA on SQuAD 2.0:
python -m models.bert --dataset SQuAD --model bert-base-uncased --num-workers 4 --batch-size 16 --epochs 3

AdapterPooler on BERT-QA SQuAD 2.0 embeddings
python -m models.adapter_pooler --dataset SQuADH5 --model AP --num-workers 0 --batch-size 16 --epochs 3 --lr 1e-3

SST on BERT for SequenceClassification
python -m models.sst --dataset SST --model SST --num-workers 0 --batch-size 16 --epochs 1 --lr 0.001
# i am only predicting 0s for some reason




results to add: 

bert qa
epoch  exact,    f1,      loss dev
 1      71.3131  74.6153  1.0209
 2      71.8689  75.5646  1.1189
 3      73.460   76.893   1.237

adapter pooler
exact,    f1,      loss dev
69.013  71.9318    1.05


