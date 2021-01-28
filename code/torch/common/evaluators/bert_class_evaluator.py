# packages
import time, os, sys, datetime
sys.path.append("C:/BERTVision/code/torch")
from data.bert_processors.sst_processor import SSTProcessor, Tokenize_Transform
from utils.collate import collate_SST
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from sklearn import metrics
import warnings
# Suppress warnings from sklearn.metrics
warnings.filterwarnings('ignore')

# tokenizer
#tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

class BertClassEvaluator(object):
    '''
    This class handles the evaluation of classification models with BERT
    architecture.

    Parameters
    ----------
    model : object
        A huggingface QuestionAnswering BERT transformer

    processor: object
        A Torch Dataset processor that emits data

    args: object
        A argument parser object; see args.py

    Operations
    -------
    This trainer:
        (1) Trains the weights
        (2) Generates dev set loss
        (3) Creates start and end logits and collects their original index for scoring
        (4) Writes their results and saves the file as a checkpoint

    '''
    def __init__(self, model, tokenizer, processor, args):
        # pull in init
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor

        # set placeholders for evaluation metrics
        self.dev_loss = 0
        self.nb_dev_steps = 0
        self.nb_dev_examples = 0

    def get_loss(self):
        '''
        This function prepares the data and handles the validation set testing.
        '''
        # instantiate dev set processor
        self.dev_examples = self.processor(type='dev',
                          is_multilabel=False,
                          transform=Tokenize_Transform(tokenizer=self.tokenizer))

        # craete dev set data loader
        dev_dataloader = DataLoader(self.dev_examples,
                                    batch_size=self.args.batch_size,
                                    shuffle=True,
                                    num_workers=self.args.num_workers,
                                    drop_last=False,
                                    collate_fn=collate_SST)

        # set the model to evaluation
        self.model.eval()

        # store results
        predicted_labels, target_labels = list(), list()

        # for each batch of data,
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Evaluating")):
            # send it to the GPU
            input_ids, attn_mask, token_type_ids, labels = (
                batch['input_ids'].to(self.args.device),
                batch['attention_mask'].to(self.args.device),
                batch['token_type_ids'].to(self.args.device),
                batch['labels'].to(self.args.device)
            )

            # forward
            with torch.no_grad():
                out = self.model(
                                 input_ids=input_ids,
                                 attention_mask=attn_mask,
                                 token_type_ids=None,
                                 labels=labels
                                 )

                probas = F.log_softmax(out.logits, dim=1)
                _, pred_labels = torch.max(probas, 1)
            # loss
            if self.args.n_gpu > 1:
                loss = out.loss.mean()
            else:
                loss = out.loss

            if self.args.is_multilabel:
                self.predicted_labels.extend(F.sigmoid(out.logits).round().long().cpu().detach().numpy())
                self.target_labels.extend(labels.cpu().detach().numpy())
            else:

                predicted_labels.extend(pred_labels.cpu().detach().numpy())
                target_labels.extend(labels.cpu().detach().numpy())

            # loss metrics
            self.dev_loss += loss.item()
            self.nb_dev_examples += input_ids.size(0)
            self.nb_dev_steps += 1

        # metrics
        predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
        print(predicted_labels)
        print('\n')
        print(target_labels)
        accuracy = metrics.accuracy_score(target_labels, predicted_labels)
        precision = metrics.precision_score(target_labels, predicted_labels, average='micro')
        recall = metrics.recall_score(target_labels, predicted_labels, average='micro')
        f1 = metrics.f1_score(target_labels, predicted_labels, average='micro')
        avg_loss = self.dev_loss / self.nb_dev_steps

        return accuracy, precision, recall, f1, avg_loss

#
