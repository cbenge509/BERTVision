{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "deluxe-airfare",
   "metadata": {},
   "outputs": [],
   "source": [
    "import transformers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "alpine-belle",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from data.bert_processors.processors import QQPairs\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "personal-tower",
   "metadata": {},
   "outputs": [],
   "source": [
    "qqp = QQPairs('train')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "charitable-irish",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n",
      "10000\n",
      "20000\n",
      "30000\n",
      "40000\n",
      "50000\n",
      "60000\n",
      "70000\n",
      "80000\n",
      "90000\n",
      "100000\n",
      "110000\n",
      "120000\n",
      "130000\n",
      "140000\n",
      "150000\n",
      "160000\n",
      "170000\n",
      "180000\n",
      "190000\n",
      "200000\n",
      "210000\n",
      "220000\n",
      "230000\n",
      "240000\n",
      "250000\n",
      "260000\n",
      "270000\n",
      "280000\n",
      "290000\n",
      "300000\n",
      "310000\n",
      "320000\n",
      "330000\n",
      "340000\n",
      "350000\n",
      "360000\n"
     ]
    }
   ],
   "source": [
    "length = []\n",
    "for i in range(len(qqp)):\n",
    "    length.append(qqp[i]['attention_mask'].numpy().sum())\n",
    "    if not i % 10000:\n",
    "        print(i)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "solar-press",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "334"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "max(length)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "industrial-punch",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([2.9424e+05, 6.5168e+04, 3.9910e+03, 3.6800e+02, 4.2000e+01,\n",
       "        1.4000e+01, 3.0000e+00, 3.0000e+00, 0.0000e+00, 1.7000e+01]),\n",
       " array([  6. ,  38.8,  71.6, 104.4, 137.2, 170. , 202.8, 235.6, 268.4,\n",
       "        301.2, 334. ]),\n",
       " <BarContainer object of 10 artists>)"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "plt.hist(length)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "numerical-yukon",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "54.0"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.percentile(length, 95\n",
    "             )"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "subject-marks",
   "metadata": {},
   "source": [
    "### Manually evaluating model and dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "prescription-nightmare",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Args(object):\n",
    "    def __init__(self):\n",
    "        self.batch_size = 8\n",
    "        self.num_workers = 4\n",
    "        self.n_gpu = 1\n",
    "        self.device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "minus-enemy",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "cuda\n"
     ]
    }
   ],
   "source": [
    "from common.evaluators.bert_class_evaluator_sj import BertClassEvaluator\n",
    "from data.bert_processors.processors import QNLI\n",
    "from models.qnli.args import get_args\n",
    "from transformers import BertForSequenceClassification\n",
    "import torch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "reduced-lucas",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForSequenceClassification: ['cls.predictions.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias']\n",
      "- This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).\n",
      "- This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).\n",
      "Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.weight', 'classifier.bias']\n",
      "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<All keys matched successfully>"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model =BertForSequenceClassification.from_pretrained('bert-base-uncased',\n",
    "                                                          num_labels=2,\n",
    "                                                          output_hidden_states=True).to('cuda')\n",
    "model.load_state_dict(torch.load('C:\\w266\\data2\\checkpoints\\BERT-QNLI_epoch_1.pt'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "transsexual-neutral",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "b'Skipping line 10344: expected 4 fields, saw 5\\nSkipping line 10897: expected 4 fields, saw 5\\nSkipping line 11356: expected 4 fields, saw 5\\nSkipping line 11367: expected 4 fields, saw 5\\nSkipping line 16599: expected 4 fields, saw 5\\nSkipping line 17114: expected 4 fields, saw 5\\nSkipping line 23153: expected 4 fields, saw 5\\nSkipping line 25672: expected 4 fields, saw 5\\nSkipping line 31107: expected 4 fields, saw 5\\nSkipping line 31359: expected 4 fields, saw 5\\nSkipping line 31402: expected 4 fields, saw 5\\nSkipping line 32555: expected 4 fields, saw 5\\nSkipping line 38524: expected 4 fields, saw 5\\nSkipping line 46338: expected 4 fields, saw 5\\nSkipping line 47889: expected 4 fields, saw 5\\nSkipping line 56759: expected 4 fields, saw 5\\nSkipping line 56850: expected 4 fields, saw 5\\nSkipping line 56919: expected 4 fields, saw 5\\nSkipping line 57514: expected 4 fields, saw 5\\nSkipping line 67155: expected 4 fields, saw 5\\nSkipping line 75061: expected 4 fields, saw 5\\nSkipping line 75721: expected 4 fields, saw 5\\nSkipping line 76275: expected 4 fields, saw 5\\nSkipping line 81073: expected 4 fields, saw 5\\nSkipping line 81535: expected 4 fields, saw 5\\nSkipping line 82386: expected 4 fields, saw 5\\nSkipping line 83818: expected 4 fields, saw 5\\nSkipping line 84674: expected 4 fields, saw 5\\nSkipping line 88965: expected 4 fields, saw 5\\nSkipping line 90380: expected 4 fields, saw 5\\nSkipping line 91441: expected 4 fields, saw 5\\nSkipping line 91640: expected 4 fields, saw 5\\nSkipping line 92513: expected 4 fields, saw 5\\nSkipping line 102152: expected 4 fields, saw 5\\n'\n",
      "Evaluating:  10%|█████████████▊                                                                                                                        | 1329/12889 [02:23<19:11, 10.04it/s]"
     ]
    }
   ],
   "source": [
    "e = BertClassEvaluator(model, QNLI, Args()).get_loss('train')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
