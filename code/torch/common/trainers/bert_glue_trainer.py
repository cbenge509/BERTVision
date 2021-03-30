# packages
import os, sys, datetime
sys.path.append("C:/BERTVision/code/torch")
from data.bert_processors.processors import Tokenize_Transform
from common.evaluators.bert_glue_evaluator import BertGLUEEvaluator
from utils.collate import collate_BERT
from torch.cuda.amp import autocast
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tqdm.notebook import trange
import numpy as np
import json

class BertGLUETrainer(object):
    '''
    This class handles the training of classification models with BERT
    architecture.

    Parameters
    ----------
    model : object
        A HuggingFace Classification BERT transformer

    tokenizer: object
        A HuggingFace tokenizer that fits the HuggingFace transformer

    optimizer: object
        A compatible Torch optimizer

    processor: object
        A Torch Dataset processor that emits data

    scheduler: object
        The learning rate decreases linearly from the initial lr set

    args: object
        A argument parser object; see args.py

    scaler: object
        A gradient scaler object to use FP16

    Operations
    -------
    This trainer:
        (1) Trains the weights
        (2) Generates dev set loss
        (3) Creates start and end logits and collects their original index for scoring
        (4) Writes their results and saves the file as a checkpoint

    '''
    def __init__(self, model, optimizer, processor, scheduler, args, scaler, logger, save = True):
        # pull in objects
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.processor = processor
        self.scheduler = scheduler
        self.scaler = scaler
        self.logger = logger
        self.save = save

        # specify training data set
        self.train_examples = self.processor(type='train', transform=Tokenize_Transform(self.args, self.logger))

        # declare progress
        self.logger.info(f"Initializing {self.args.model}-train with {self.args.max_seq_length} token length")

        # create a timestamp for the checkpoints
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # create a location to save the files
        make_path = os.path.join(self.args.save_path, self.args.checkpoint, self.args.model)
        os.makedirs(make_path, exist_ok=True)
        self.snapshot_path = os.path.join(self.args.save_path, self.args.checkpoint, self.args.model, '%s.pt' % timestamp)

        # determine the number of optimization steps
        self.num_train_optimization_steps = int(
            len(self.train_examples) / args.batch_size) * args.epochs

        # create placeholders for model metrics and early stopping if desired
        self.iterations, self.nb_tr_steps, self.tr_loss = 0, 0, 0
        self.best_dev_f1, self.unimproved_iters, self.dev_loss, self.dev_acc = 0, 0, np.inf, -np.inf
        self.early_stop = False

    def train_epoch(self, train_dataloader):
        # set the model to train
        self.model.train()
        # pull data from data loader
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            # and sent it to the GPU
            input_ids, attn_mask, token_type_ids, labels, idxs = (
                batch['input_ids'].to(self.args.device),
                batch['attention_mask'].to(self.args.device),
                batch['token_type_ids'].to(self.args.device),
                batch['labels'].to(self.args.device),
                batch['idx'].to(self.args.device)
            )

            # FP16
            with autocast():
                # forward
                out = self.model(
                                 input_ids=input_ids,
                                 attention_mask=attn_mask,
                                 token_type_ids=token_type_ids,
                                 labels=labels
                                 )

            # loss
            if self.args.n_gpu > 1:
                loss = out.loss.mean()
            else:
                loss = out.loss

            # backward
            self.scaler.scale(out.loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()

            # update metrics
            self.tr_loss += loss.item()
            self.nb_tr_steps += 1

        # gen. loss
        avg_loss = self.tr_loss / self.nb_tr_steps

        # print end of trainig results
        self.logger.info(f"Training complete! Loss: {avg_loss}")
        return avg_loss

    def train(self):
        '''
        This function handles the entirety of the training, dev, and scoring.
        '''
        # tell the user general metrics
        self.logger.info(f"Number of examples: {len(self.train_examples)}")
        self.logger.info(f"Batch size: {self.args.batch_size}")
        self.logger.info(f"Number of optimization steps: {self.num_train_optimization_steps}")

        # instantiate dataloader
        train_dataloader = DataLoader(self.train_examples,
                                      batch_size=self.args.batch_size,
                                      shuffle=True,
                                      num_workers=self.args.num_workers,
                                      drop_last=False,
                                      collate_fn=collate_BERT)

        # for each epoch
        from collections import defaultdict
        losses = defaultdict(list)
        for epoch in trange(int(self.args.epochs), desc="Epoch"):

            if any([self.args.model == 'SST',
                    self.args.model == 'MSR',
                    self.args.model == 'RTE',
                    self.args.model == 'QNLI',
                    self.args.model == 'QQP',
                    self.args.model == 'WNLI'
                    ]):

                # train
                losses['train_loss'].append(self.train_epoch(train_dataloader))
                # get dev loss
                dev_acc, dev_precision, dev_recall, dev_f1, dev_loss = BertGLUEEvaluator(self.model, self.processor, self.args, self.logger).get_loss(type='dev')
                # print validation results
                self.logger.info("Epoch {0: d}, Dev/Acc {1: 0.3f}, Dev/Pr. {2: 0.3f}, Dev/Re. {3: 0.3f}, Dev/F1 {4: 0.3f}, Dev/Loss {5: 0.3f}",
                                 epoch+1, dev_acc, dev_precision, dev_recall, dev_f1, dev_loss)
                losses['dev_acc'].append(dev_acc)
                # update validation results
                if dev_acc > self.dev_acc:
                    self.unimproved_iters = 0
                    self.dev_acc = dev_acc
                    if self.save:
                        torch.save(self.model, self.snapshot_path)

                else:
                    # stop training with early stopping
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.args.patience:
                        self.early_stop = True
                        self.logger.info(f"Early Stopping. Epoch: {epoch}, Best Dev Acc: {self.dev_acc}")
                        #break


            elif any([self.args.model == 'CoLA']):

                # train
                self.train_epoch(train_dataloader)
                # get dev loss
                matthews, dev_loss = BertGLUEEvaluator(self.model, self.processor, self.args, self.logger).get_loss(type='dev')
                # print validation results
                self.logger.info("Epoch {0: d}, Dev/Matthews {1: 0.3f}, Dev/Loss {2: 0.3f}",
                                 epoch+1, matthews, dev_loss)

                # update validation results
                if dev_loss < self.dev_loss:
                    self.unimproved_iters = 0
                    self.dev_loss = dev_loss
                    if self.save:
                        torch.save(self.model, self.snapshot_path)

                else:
                    # stop training with early stopping
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.args.patience:
                        self.early_stop = True
                        self.logger.info(f"Early Stopping. Epoch: {epoch}, Best Dev Loss: {self.dev_loss}")
                        #break


            elif any([self.args.model == 'STSB']):

                # train
                self.train_epoch(train_dataloader)
                # get dev loss
                pearson, spearman, dev_loss = BertGLUEEvaluator(self.model, self.processor, self.args, self.logger).get_loss(type='dev')
                # print validation results
                self.logger.info("Epoch {0: d}, Dev/Pearson {1: 0.3f}, Dev/Spearman {2: 0.3f}, Dev/Loss {3: 0.3f}",
                                 epoch+1, pearson, spearman, dev_loss)

                # update validation results
                if dev_loss < self.dev_loss:
                    self.unimproved_iters = 0
                    self.dev_loss = dev_loss
                    if self.save:
                        torch.save(self.model, self.snapshot_path)

                else:
                    # stop training with early stopping
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.args.patience:
                        self.early_stop = True
                        self.logger.info(f"Early Stopping. Epoch: {epoch}, Best Dev Loss: {self.dev_loss}")
                        #break


            elif any([self.args.model == 'MNLI']):

                # train
                self.train_epoch(train_dataloader)
                # matched
                dev_acc, dev_precision, dev_recall, dev_f1, dev_loss = BertGLUEEvaluator(self.model, self.processor, self.args, self.logger).get_loss(type='dev_matched')
                # print validation results
                self.logger.info("Epoch {0: d}, Dev/Acc {1: 0.3f}, Dev/Pr. {1: 0.3f}, Dev/Re. {1: 0.3f}, Dev/F1 {1: 0.3f}, Dev/Loss {1: 0.3f}",
                                 epoch+1, dev_acc, dev_precision, dev_recall, dev_f1, dev_loss)
                # update validation results
                if dev_loss < self.dev_loss:
                    self.unimproved_iters = 0
                    self.dev_loss = dev_loss
                    if self.save:
                        torch.save(self.model, self.snapshot_path)
                else:
                    # stop training with early stopping
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.args.patience:
                        self.early_stop = True
                        self.logger.info(f"Early Stopping. Epoch: {epoch}, Best Dev Loss: {self.dev_loss}")
                        break

                # matched
                dev_acc, dev_precision, dev_recall, dev_f1, dev_loss = BertGLUEEvaluator(self.model, self.processor, self.args, self.logger).get_loss(type='dev_mismatched')
                # print validation results
                self.logger.info("Epoch {0: d}, Dev/Acc {1: 0.3f}, Dev/Pr. {1: 0.3f}, Dev/Re. {1: 0.3f}, Dev/F1 {1: 0.3f}, Dev/Loss {1: 0.3f}",
                                 epoch+1, dev_acc, dev_precision, dev_recall, dev_f1, dev_loss)

                # update validation results
                if dev_loss < self.dev_loss:
                    self.unimproved_iters = 0
                    self.dev_loss = dev_loss
                    if self.save:
                        torch.save(self.model, self.snapshot_path)

                else:
                    # stop training with early stopping
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.args.patience:
                        self.early_stop = True
                        self.logger.info(f"Early Stopping. Epoch: {epoch}, Best Dev Loss: {self.dev_loss}")
                        break

        if self.args.file:
            with open(self.args.file, 'w') as fp:
                json.dump(dict(losses), fp)
            print(losses)

#
