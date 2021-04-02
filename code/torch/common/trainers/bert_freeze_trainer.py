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
import copy


class BertFreezeTrainer(object):
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
    def __init__(self, model, optimizer, processor, scheduler, args, kwargs, scaler, logger):
        # pull in objects
        self.args = args
        self.kwargs = kwargs
        self.model = model
        self.optimizer = optimizer
        self.processor = processor
        self.scheduler = scheduler
        self.scaler = scaler
        self.logger = logger

        # shard the large datasets:
        if any([self.args.model == 'QQP',
                self.args.model == 'QNLI',
                self.args.model == 'MNLI',
                self.args.model == 'SST'
                ]):
            # turn on sharding
            self.train_examples = self.processor(type='train', transform=Tokenize_Transform(self.args, self.logger), shard=True, kwargs=self.kwargs)

        else:
            # create the usual processor
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
        self.best_dev_f1, self.unimproved_iters, self.dev_loss = 0, 0, np.inf
        self.early_stop = False

        # save a copy of initial bert params
        initial_weights = copy.deepcopy(self.model.state_dict())

        # retrieve a freeze value
        #self.freeze_p = np.random.uniform(0.05, 0.95)
        self.freeze_p = self.args.freeze_p
        #self.freeze_p = 1.0

        # declare progress
        self.logger.info(f"Freezing this % of params now: {self.freeze_p}")
        np.random.seed(seed=self.args.seed)

        # randomly find weights to take, but take in this condition
        inject = self.args.inject
        reject = self.args.reject


        # created mask to load weights
        self.locked_masks = {
                        name: (
                            torch.tensor(np.random.choice([False, True],
                                                          size=torch.numel(weight),
                                                          p=[self.freeze_p, (1-self.freeze_p)])
                                         .reshape(weight.shape))

                            if any(weight in name for weight in inject)
                            and not any(weight in name for weight in reject) else
                            torch.tensor(np.random.choice([False, True],
                                                          size=torch.numel(weight),
                                                          p=[0.0, 1.0])
                                         .reshape(weight.shape))
                          )
                        for name, weight in initial_weights.items()
                        }

        # load trained bert model state
        # RTE
        #self.model = torch.load('C:\\BERTVision\\code\\torch\\model_checkpoints\\bert-base-uncased\\RTE\\2021-03-11_20-14-05.pt')
        # CoLA
        self.model = torch.load('C:\\BERTVision\\code\\torch\\model_checkpoints\\bert-base-uncased\\CoLA\\2021-03-28_14-25-35.pt')

        # create a copy of the weights
        trained_weights = copy.deepcopy(self.model.state_dict())

        # create a new model state
        result = {}
        # for each key
        for key, value in initial_weights.items():
            # add the key
            result[key] = []
            # if False, replace initial value with trained value
            result[key] = initial_weights[key].cuda().where(self.locked_masks[key].cuda(), trained_weights[key].cuda())

        # load the model with the new mix of weights
        self.model.load_state_dict(result)

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
            # zero gradients of interest
            for name, weight in self.model.named_parameters():
                if weight.grad is not None and name in self.locked_masks:
                    weight.grad[self.locked_masks[name]] = 0
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
        self.logger.info(f"Batch size: {len(self.train_examples)}")
        self.logger.info(f"Number of optimization steps: {self.num_train_optimization_steps}")

        # storage containers for training metrics
        self.epoch_loss = list()
        self.epoch_metric = list()
        self.epoch = list()
        self.epoch_freeze_p = list()

        # instantiate dataloader
        train_dataloader = DataLoader(self.train_examples,
                                      batch_size=self.args.batch_size,
                                      shuffle=True,
                                      num_workers=self.args.num_workers,
                                      drop_last=False,
                                      collate_fn=collate_BERT)
        # for each epoch
        if any([self.args.model == 'SST',
                self.args.model == 'MSR',
                self.args.model == 'RTE',
                self.args.model == 'QNLI',
                self.args.model == 'QQP',
                self.args.model == 'WNLI'
                ]):

            for epoch in trange(int(self.args.epochs), desc="Epoch"):
                # train
                #self.train_epoch(train_dataloader)  # temporarily skip; go straight to evaluation
                # get dev loss
                dev_acc, dev_precision, dev_recall, dev_f1, dev_loss, mean_preds = BertGLUEEvaluator(self.model, self.processor, self.args, self.logger).get_loss(type='dev')
                # print validation results
                self.logger.info("Epoch {0: d}, Dev/Acc {1: 0.3f}, Dev/Pr. {2: 0.3f}, Dev/Re. {3: 0.3f}, Dev/F1 {4: 0.3f}, Dev/Loss {5: 0.3f}",
                                 epoch+1, dev_acc, dev_precision, dev_recall, dev_f1, dev_loss)

                self.epoch_loss.append(dev_loss)
                self.epoch_metric.append(dev_acc)
                self.epoch.append(epoch+1)
                self.epoch_freeze_p.append(self.freeze_p)

            return self.epoch_loss, self.epoch_metric, self.epoch, self.epoch_freeze_p, mean_preds


        elif any([self.args.model == 'CoLA']):

            for epoch in trange(int(self.args.epochs), desc="Epoch"):
                # train
                #self.train_epoch(train_dataloader)  # temporarily skip; go straight to evaluation
                # get dev loss
                matthews, dev_loss, mean_preds = BertGLUEEvaluator(self.model, self.processor, self.args, self.logger).get_loss(type='dev')
                # print validation results
                self.logger.info("Epoch {0: d}, Dev/Matthews {1: 0.3f}, Dev/Loss {2: 0.3f}",
                                 epoch+1, matthews, dev_loss)

                self.epoch_loss.append(dev_loss)
                self.epoch_metric.append(matthews)
                self.epoch.append(epoch+1)
                self.epoch_freeze_p.append(self.freeze_p)

            return self.epoch_loss, self.epoch_metric, self.epoch, self.epoch_freeze_p, mean_preds


        elif any([self.args.model == 'STSB']):

            for epoch in trange(int(self.args.epochs), desc="Epoch"):
                # train
                self.train_epoch(train_dataloader)
                # get dev loss
                pearson, spearman, dev_loss = BertGLUEEvaluator(self.model, self.processor, self.args, self.logger).get_loss(type='dev')
                # print validation results
                self.logger.info("Epoch {0: d}, Dev/Pearson {1: 0.3f}, Dev/Spearman {2: 0.3f}, Dev/Loss {3: 0.3f}",
                                 epoch+1, pearson, spearman, dev_loss)

                # take the average of the two tests
                avg_corr = (pearson + spearman) / 2

                self.epoch_loss.append(dev_loss)
                self.epoch_metric.append(avg_corr)
                self.epoch.append(epoch+1)
                self.epoch_freeze_p.append(self.freeze_p)

            return self.epoch_loss, self.epoch_metric, self.epoch, self.epoch_freeze_p


        elif any([self.args.model == 'MNLI']):

            for epoch in trange(int(self.args.epochs), desc="Epoch"):
                # train
                self.train_epoch(train_dataloader)
                # matched
                dev_acc1, dev_precision, dev_recall, dev_f1, dev_loss1 = BertGLUEEvaluator(self.model, self.processor, self.args, self.logger).get_loss(type='dev_matched')
                # print validation results
                self.logger.info("Epoch {0: d}, Dev/Acc {1: 0.3f}, Dev/Pr. {1: 0.3f}, Dev/Re. {1: 0.3f}, Dev/F1 {1: 0.3f}, Dev/Loss {1: 0.3f}",
                                 epoch+1, dev_acc1, dev_precision, dev_recall, dev_f1, dev_loss1)


                # matched
                dev_acc2, dev_precision, dev_recall, dev_f1, dev_loss2 = BertGLUEEvaluator(self.model, self.processor, self.args, self.logger).get_loss(type='dev_mismatched')
                # print validation results
                self.logger.info("Epoch {0: d}, Dev/Acc {1: 0.3f}, Dev/Pr. {1: 0.3f}, Dev/Re. {1: 0.3f}, Dev/F1 {1: 0.3f}, Dev/Loss {1: 0.3f}",
                                 epoch+1, dev_acc2, dev_precision, dev_recall, dev_f1, dev_loss2)

                # compute average loss
                dev_loss = (dev_loss1 + dev_loss2) / 2
                # compute average acc
                dev_acc = (dev_acc1 + dev_acc2) / 2

                self.epoch_loss.append(dev_loss)
                self.epoch_metric.append(dev_acc)
                self.epoch.append(epoch+1)
                self.epoch_freeze_p.append(self.freeze_p)

            return self.epoch_loss, self.epoch_metric, self.epoch, self.epoch_freeze_p

#
