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

class H5_GLUE_Trainer(object):
    '''
    This class handles the training of 1-epoch tuned QA embeddings from BERT

    Parameters
    ----------
    model : object
        A compression model; see compress_utils.py

    criterion : loss function
        A loss function

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
    def __init__(self, model, criterion, optimizer, processor, scheduler, args, scaler, logger):
        # pull in objects
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.processor = processor
        self.scheduler = scheduler
        self.scaler = scaler
        self.logger = logger

        # specify training data set
        self.train_examples = processor(type='train', args=self.args)

        # create a timestamp for the checkpoints
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # create a location to save the files
        self.snapshot_path = os.path.join(args.save_path, args.checkpoint, args.model, '%s.pt' % timestamp)
        self.make_path = os.path.join(self.args.save_path, self.args.checkpoint, self.args.model)
        os.makedirs(self.make_path, exist_ok=True)

        # determine the number of optimization steps
        self.num_train_optimization_steps = int(
            len(self.train_examples) / args.batch_size) * args.epochs

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

            # check model
            if any([self.args.model == 'AP_SST',
                    self.args.model == 'AP_MSR',
                    self.args.model == 'AP_RTE',
                    self.args.model == 'AP_QNLI',
                    self.args.model == 'AP_QQPairs',
                    self.args.model == 'AP_WNLI'
                    ]):

                # train
                self.train_epoch(self.criterion, train_dataloader)
                # get metrics
                dev_acc, dev_precision, dev_recall, dev_f1, dev_loss = H5_GLUE_Evaluator(self.model, self.criterion, self.processor, self.args, self.logger).get_loss(type='dev')
                # print validation results
                self.logger.info("Epoch {0: d}, Dev/Acc {1: 0.3f}, Dev/Pr. {2: 0.3f}, Dev/Re. {3: 0.3f}, Dev/F1 {4: 0.3f}, Dev/Loss {5: 0.3f}",
                                 epoch+1, dev_acc, dev_precision, dev_recall, dev_f1, dev_loss)

                # update validation results
                if dev_loss < self.dev_loss:
                    self.unimproved_iters = 0
                    self.dev_loss = dev_loss
                    torch.save(self.model, self.snapshot_path)

                else:
                    # stop training with early stopping
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.args.patience:
                        self.early_stop = True
                        self.logger.info(f"Early Stopping. Epoch: {epoch}, Best Dev Loss: {self.dev_loss}")
                        break

            # get dev loss
            elif all([self.args.num_labels > 1, self.args.model == 'AP_MNLI']):

                # train
                self.train_epoch(self.criterion, train_dataloader)
                # get metrics
                dev_acc, dev_precision, dev_recall, dev_f1, dev_loss = H5_GLUE_Evaluator(self.model, self.criterion, self.processor, self.args, self.logger).get_loss(type='dev_matched')
                # print validation results
                self.logger.info("Epoch {0: d}, Dev/Acc {1: 0.3f}, Dev/Pr. {2: 0.3f}, Dev/Re. {3: 0.3f}, Dev/F1 {4: 0.3f}, Dev/Loss {5: 0.3f}",
                                 epoch+1, dev_acc, dev_precision, dev_recall, dev_f1, dev_loss)

                # update validation results
                if dev_loss < self.dev_loss:
                    self.unimproved_iters = 0
                    self.dev_loss = dev_loss
                    torch.save(self.model, self.snapshot_path)

                else:
                    # stop training with early stopping
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.args.patience:
                        self.early_stop = True
                        self.logger.info(f"Early Stopping. Epoch: {epoch}, Best Dev Loss: {self.dev_loss}")
                        break

                # get metrics
                dev_acc, dev_precision, dev_recall, dev_f1, dev_loss = H5_GLUE_Evaluator(self.model, self.criterion, self.processor, self.args, self.logger).get_loss(type='dev_mismatched')
                # print validation results
                self.logger.info("Epoch {0: d}, Dev/Acc {1: 0.3f}, Dev/Pr. {2: 0.3f}, Dev/Re. {3: 0.3f}, Dev/F1 {4: 0.3f}, Dev/Loss {5: 0.3f}",
                                 epoch+1, dev_acc, dev_precision, dev_recall, dev_f1, dev_loss)

                # update validation results
                if dev_loss < self.dev_loss:
                    self.unimproved_iters = 0
                    self.dev_loss = dev_loss
                    torch.save(self.model, self.snapshot_path)

                else:
                    # stop training with early stopping
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.args.patience:
                        self.early_stop = True
                        self.logger.info(f"Early Stopping. Epoch: {epoch}, Best Dev Loss: {self.dev_loss}")
                        break

            elif all([self.args.num_labels == 1 and self.args.model == 'AP_STSB']):

                # get metrics
                pearson, spearman, dev_loss = H5_GLUE_Evaluator(self.model, self.criterion, self.processor, self.args, self.logger).get_loss(type='dev')

                # print validation results
                self.logger.info("Epoch {0: d}, Dev/Pearson {1: 0.3f}, Dev/Spearman {2: 0.3f}, Dev/Loss {3: 0.3f}",
                                 epoch+1, pearson, spearman, dev_loss)

                # update validation results
                if dev_loss < self.dev_loss:
                    self.unimproved_iters = 0
                    self.dev_loss = dev_loss
                    torch.save(self.model, self.snapshot_path)

                else:
                    # stop training with early stopping
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.args.patience:
                        self.early_stop = True
                        self.logger.info(f"Early Stopping. Epoch: {epoch}, Best Dev Loss: {self.dev_loss}")
                        break

            elif self.args.model == 'AP_COLA':

                # get metrics
                matthews, dev_loss = H5_GLUE_Evaluator(self.model, self.criterion, self.processor, self.args, self.logger).get_loss(type='dev')
                # print validation results
                self.logger.info("Epoch {0: d}, Dev/Matthews {1: 0.3f}, Dev/Loss {2: 0.3f}",
                                 epoch+1, matthews, dev_loss)

                # update validation results
                if dev_loss < self.dev_loss:
                    self.unimproved_iters = 0
                    self.dev_loss = dev_loss
                    torch.save(self.model, self.snapshot_path)

                else:
                    # stop training with early stopping
                    self.unimproved_iters += 1
                    if self.unimproved_iters >= self.args.patience:
                        self.early_stop = True
                        self.logger.info(f"Early Stopping. Epoch: {epoch}, Best Dev Loss: {self.dev_loss}")
                        break

#
