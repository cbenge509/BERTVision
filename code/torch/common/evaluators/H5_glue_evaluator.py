# packages
import os, sys, datetime, time
sys.path.append("C:/BERTVision/code/torch")
from utils.collate import collate_H5_GLUE
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
import warnings
from scipy.stats import pearsonr, spearmanr
# Suppress warnings from sklearn.metrics
warnings.filterwarnings('ignore')

class H5_GLUE_Evaluator(object):
    '''
    This class handles the evaluation of 1-epoch tuned QA embeddings from BERT

    Parameters
    ----------
    model : object
        A huggingface QuestionAnswering BERT transformer

    criterion : loss function
        A loss function

    processor: object
        A Torch Dataset processor that emits data

    args: object
        A argument parser object; see args.py

    '''
    def __init__(self, model, criterion, processor, args, logger):
        # pull in init
        self.args = args
        self.model = model
        self.criterion = criterion
        self.processor = processor
        self.logger = logger

        # set placeholders for evaluation metrics
        self.dev_loss = 0
        self.nb_dev_steps = 0
        self.nb_dev_examples = 0

    def get_loss(self, type):
        '''
        This function prepares the data and handles the validation set testing.
        '''
        # create string flag
        self.type = type
        # instantiate dev set processor
        self.dev_examples = self.processor(type=self.type, args=self.args)

        # craete dev set data loader
        dev_dataloader = DataLoader(self.dev_examples,
                                    batch_size=self.args.batch_size,
                                    shuffle=True,
                                    num_workers=self.args.num_workers,
                                    drop_last=False,
                                    collate_fn=collate_H5_GLUE)

        # set the model to evaluation
        self.model.eval()

        # store results
        predicted_labels, target_labels = list(), list()

        # for each batch of data,
        self.logger.info(f"Generating metrics")
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Evaluating")):
            # send it to the GPU
            embeddings, labels, indices = (
                batch['embeddings'].to(self.args.device),
                batch['labels'].to(self.args.device),
                batch['idx'].to(self.args.device)
            )

            # forward
            with torch.no_grad():
                logits = self.model(embeddings)

                # get loss
                if self.args.num_labels == 1:
                    loss = self.criterion(logits.view(-1), labels.view(-1))
                else:
                    loss = self.criterion(logits, labels)

                # preds
                if self.args.num_labels > 1:
                    pred = logits.max(1)[1]  # get the index of the max log-probability
                else:
                    pred = logits

            # multigpu loss
            if self.args.n_gpu > 1:
                raise NotImplementedError

            # store y and y-hat
            predicted_labels.extend(pred.cpu().detach().numpy().flatten())
            target_labels.extend(labels.cpu().detach().numpy())

            # loss metrics
            self.dev_loss += loss.item()
            self.nb_dev_examples += embeddings.size(0)
            self.nb_dev_steps += 1

        # metrics
        predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
        avg_loss = self.dev_loss / self.nb_dev_steps

        if any([self.args.model == 'AP_SST',
                self.args.model == 'AP_MSR',
                self.args.model == 'AP_RTE',
                self.args.model == 'AP_QNLI',
                self.args.model == 'AP_QQP',
                self.args.model == 'AP_MNLI',
                self.args.model == 'AP_WNLI',
                ]):

            predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
            accuracy = metrics.accuracy_score(target_labels, predicted_labels)
            precision = metrics.precision_score(target_labels, predicted_labels, average='micro')
            recall = metrics.recall_score(target_labels, predicted_labels, average='micro')
            f1 = metrics.f1_score(target_labels, predicted_labels, average='micro')
            return accuracy, precision, recall, f1, avg_loss

        elif any([self.args.model == 'AP_COLA']):

            matthew1 = metrics.matthews_corrcoef(target_labels, predicted_labels)
            return matthew1, avg_loss

        elif any([self.args.model == 'AP_STSB']):

            pearson = pearsonr(predicted_labels, target_labels)[0]
            spearman = spearmanr(predicted_labels, target_labels)[0]
            return pearson, spearman, avg_loss


#
