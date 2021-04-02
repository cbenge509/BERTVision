# packages
import os, sys, datetime
sys.path.append("C:/BERTVision/code/torch")
from common.evaluators.H5_glue_evaluator import H5_GLUE_Evaluator
from utils.collate import collate_H5_GLUE
from torch.cuda.amp import autocast
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tqdm.notebook import trange
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup

class H5SearchTrainer(object):
    '''
    This class handles the training of classification models with BERT
    architecture.

    Parameters
    ----------
    model : object
        A HuggingFace Classification BERT transformer

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
    def __init__(self, model, processor, criterion, args, scaler, logger):
        # pull in objects
        self.args = args
        self.model = model
        self.processor = processor
        self.criterion = criterion
        self.scaler = scaler
        self.logger = logger

        # shard the large datasets:
        if any([self.args.model == 'AP_QQP',
                self.args.model == 'AP_QNLI',
                self.args.model == 'AP_MNLI',
                self.args.model == 'AP_SST',
                self.args.model == 'AP_MSR'
                ]):
            # turn on sharding
            self.train_examples = self.processor(type='train', args=self.args, shard=True, seed=args.seed)

        else:
            # create the usual processor
            self.train_examples = self.processor(type='train', args=self.args)

        # determine the number of optimization steps
        self.num_train_optimization_steps = int(
            len(self.train_examples) / args.batch_size) * args.epochs

        # set optimizer
        param_optimizer = list(model.named_parameters())

        # exclude these from regularization
        no_decay = ['bias']
        # give l2 regularization to any parameter that is not named after no_decay list
        # give no l2 regulariation to any bias parameter or layernorm bias/weight
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.l2},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        # set optimizer
        self.optimizer = AdamW(optimizer_grouped_parameters,
                                  lr=args.lr,
                                  correct_bias=False,
                                  weight_decay=args.l2)

        # set linear scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_training_steps=self.num_train_optimization_steps,
                                                    num_warmup_steps=args.warmup_proportion * self.num_train_optimization_steps)

        # create a timestamp for the checkpoints
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # create a location to save the files
        self.snapshot_path = os.path.join(args.save_path, args.checkpoint, args.model, '%s.pt' % timestamp)
        self.make_path = os.path.join(self.args.save_path, self.args.checkpoint, self.args.model)
        os.makedirs(self.make_path, exist_ok=True)

        # create placeholders for model metrics and early stopping if desired
        self.iterations, self.nb_tr_steps, self.tr_loss = 0, 0, 0
        self.best_dev_f1, self.unimproved_iters, self.dev_loss = 0, 0, np.inf
        self.pearson_score = 0
        self.early_stop = False

    def train_epoch(self, criterion, train_dataloader):
        # set the model to train
        self.model.train()
        # pull data from data loader
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            # and sent it to the GPU
            embeddings, labels, indices = (
                batch['embeddings'].to(self.args.device),
                batch['labels'].to(self.args.device),
                batch['idx'].to(self.args.device)
            )

            # FP16
            with autocast():
                # forward
                logits = self.model(embeddings)

                    # get loss
                if self.args.num_labels == 1:
                    loss = criterion(logits.view(-1), labels.view(-1))
                else:
                    loss = criterion(logits, labels)

                # multi-gpu loss
                if self.args.n_gpu > 1:
                    raise NotImplementedError

                # backward
                self.scaler.scale(loss).backward()
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
                                      collate_fn=collate_H5_GLUE)
        # for each epoch
        for epoch in trange(int(self.args.epochs), desc="Epoch"):

            if any([self.args.model == 'AP_SST',
                    self.args.model == 'AP_MSR',
                    self.args.model == 'AP_RTE',
                    self.args.model == 'AP_QNLI',
                    self.args.model == 'AP_QQP',
                    self.args.model == 'AP_WNLI'
                    ]):


                # train
                self.train_epoch(self.criterion, train_dataloader)
                # get dev loss
                dev_acc, dev_precision, dev_recall, dev_f1, dev_loss = H5_GLUE_Evaluator(self.model, self.criterion, self.processor, self.args, self.logger).get_loss(type='dev')
                # print validation results
                self.logger.info("Epoch {0: d}, Dev/Acc {1: 0.3f}, Dev/Pr. {2: 0.3f}, Dev/Re. {3: 0.3f}, Dev/F1 {4: 0.3f}, Dev/Loss {5: 0.3f}",
                                 epoch+1, dev_acc, dev_precision, dev_recall, dev_f1, dev_loss)

                return dev_acc


            elif any([self.args.model == 'AP_CoLA']):

                # train
                self.train_epoch(self.criterion, train_dataloader)
                # get dev loss
                matthews, dev_loss = H5_GLUE_Evaluator(self.model, self.criterion, self.processor, self.args, self.logger).get_loss(type='dev')
                # print validation results
                self.logger.info("Epoch {0: d}, Dev/Matthews {1: 0.3f}, Dev/Loss {2: 0.3f}",
                                 epoch+1, matthews, dev_loss)

                return matthews


            elif any([self.args.model == 'AP_STSB']):

                # train
                self.train_epoch(self.criterion, train_dataloader)
                # get dev loss
                pearson, spearman, dev_loss = H5_GLUE_Evaluator(self.model, self.criterion, self.processor, self.args, self.logger).get_loss(type='dev')
                # print validation results
                self.logger.info("Epoch {0: d}, Dev/Pearson {1: 0.3f}, Dev/Spearman {2: 0.3f}, Dev/Loss {3: 0.3f}",
                                 epoch+1, pearson, spearman, dev_loss)

                return pearson


            elif any([self.args.model == 'AP_MNLI']):

                # train
                self.train_epoch(self.criterion, train_dataloader)
                # matched
                dev_acc, dev_precision, dev_recall, dev_f1, dev_loss1 = H5_GLUE_Evaluator(self.model, self.criterion, self.processor, self.args, self.logger).get_loss(type='dev_matched')
                # print validation results
                self.logger.info("Epoch {0: d}, Dev/Acc {1: 0.3f}, Dev/Pr. {1: 0.3f}, Dev/Re. {1: 0.3f}, Dev/F1 {1: 0.3f}, Dev/Loss {1: 0.3f}",
                                 epoch+1, dev_acc, dev_precision, dev_recall, dev_f1, dev_loss1)


                # matched
                dev_acc, dev_precision, dev_recall, dev_f1, dev_loss2 = H5_GLUE_Evaluator(self.model, self.criterion, self.processor, self.args, self.logger).get_loss(type='dev_mismatched')
                # print validation results
                self.logger.info("Epoch {0: d}, Dev/Acc {1: 0.3f}, Dev/Pr. {1: 0.3f}, Dev/Re. {1: 0.3f}, Dev/F1 {1: 0.3f}, Dev/Loss {1: 0.3f}",
                                 epoch+1, dev_acc, dev_precision, dev_recall, dev_f1, dev_loss2)

                # compute average
                dev_loss = (dev_loss1 + dev_loss2) / 2

                return dev_acc

#
