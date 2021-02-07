# packages
import time, os, sys, datetime
sys.path.append("C:/BERTVision/code/torch")
from data.bert_processors.processors import COLA
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


class BertClassEvaluator(object):
    '''
    This class handles the evaluation of classification models with BERT
    architecture.

    Parameters
    ----------
    model : object
        A HuggingFace QuestionAnswering BERT transformer

    processor: object
        A Torch Dataset processor that emits data

    tokenizer: object
        A HuggingFace tokenizer that fits the HuggingFace transformer

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
    def __init__(self, model, processor, args):
        # pull in init
        self.args = args
        self.model = model
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
        self.dev_examples = self.processor(type='dev')

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
                                 token_type_ids=token_type_ids,
                                 labels=labels
                                 )

                # preds
                pred = out.logits.max(1)[1]  # get the index of the max log-probability

            # loss
            if self.args.n_gpu > 1:
                loss = out.loss.mean()
            else:
                loss = out.loss

                # in case we need something special for multi-label
            if self.args.is_multilabel:
                predicted_labels.extend(pred.cpu().detach().numpy().flatten())
                target_labels.extend(labels.cpu().detach().numpy())
            else:
                # for binary
                predicted_labels.extend(pred.cpu().detach().numpy().flatten())
                target_labels.extend(labels.cpu().detach().numpy())

            # loss metrics
            self.dev_loss += loss.item()
            self.nb_dev_examples += input_ids.size(0)
            self.nb_dev_steps += 1

        # metrics
        predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
        avg_loss = self.dev_loss / self.nb_dev_steps

        if self.args.model_name == 'COLA':
            matthew1 = metrics.matthews_corrcoef(target_labels, predicted_labels)
            return matthew1, avg_loss

    def get_loss_mnli(self, type):
        '''
        This function prepares the data and handles the validation set testing.
        '''
        self.type = type
        # instantiate dev set processor
        self.dev_examples = self.processor(type=self.type)

        # craete dev set data loader
        dev_dataloader = DataLoader(self.dev_examples,
                                    batch_size=self.args.batch_size,
                                    shuffle=True,
                                    num_workers=self.args.num_workers,
                                    drop_last=False,
                                    collate_fn=collate_H5_SST)

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
                                 token_type_ids=token_type_ids,
                                 labels=labels
                                 )

                # preds
                pred = out.logits.max(1)[1]  # get the index of the max log-probability

            # loss
            if self.args.n_gpu > 1:
                loss = out.loss.mean()
            else:
                loss = out.loss

                # in case we need something special for multi-label
            if self.args.is_multilabel:
                predicted_labels.extend(pred.cpu().detach().numpy().flatten())
                target_labels.extend(labels.cpu().detach().numpy())
            else:
                # for binary
                predicted_labels.extend(pred.cpu().detach().numpy().flatten())
                target_labels.extend(labels.cpu().detach().numpy())

            # loss metrics
            self.dev_loss += loss.item()
            self.nb_dev_examples += input_ids.size(0)
            self.nb_dev_steps += 1

        # metrics
        predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
        avg_loss = self.dev_loss / self.nb_dev_steps

        accuracy = metrics.accuracy_score(target_labels, predicted_labels)
        precision = metrics.precision_score(target_labels, predicted_labels, average='micro')
        recall = metrics.recall_score(target_labels, predicted_labels, average='micro')
        f1 = metrics.f1_score(target_labels, predicted_labels, average='micro')

        return accuracy, precision, recall, f1, avg_loss

#
