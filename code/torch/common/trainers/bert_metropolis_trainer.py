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

class BertMetropolisTrainer(object):
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

        # retrieve a freeze value between 0 and 1
        self.freeze_p = np.random.uniform(0.05, 0.95)

        # declare progress
        self.logger.info(f"Freezing this % of params now: {self.freeze_p}")
        np.random.seed(seed=1)

        # randomly find weights to take, but take in this condition
        #inject = self.args.inject
        #reject = self.args.reject
        #python -m models.pfreezing --model RTE --checkpoint bert-base-uncased --batch-size 16 --lr 2e-5 --num-labels 2 --max-seq-length 250
        self.inject_p = 0.01
        self.duration = 100000
        self.patience = 0
        self.t0_weights = copy.deepcopy(self.model.state_dict())
        self.trained_model = torch.load('C:\\BERTVision\\code\\torch\\model_checkpoints\\bert-base-uncased\\RTE\\2021-03-11_20-14-05.pt')
        self.trained_weights = copy.deepcopy(self.trained_model.state_dict())
        np.random.seed(self.args.seed)

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
        def mask_generator(inject_p, inject, reject, weights):
            inject = inject
            reject = reject
            mask = {
                            name: (
                                torch.tensor(np.random.choice([False, True],
                                                              size=torch.numel(weight),
                                                              p=[inject_p, (1-inject_p)])
                                             .reshape(weight.shape))

                                if any(weight in name for weight in inject)
                                and not any(weight in name for weight in reject) else
                                torch.tensor(np.random.choice([False, True],
                                                              size=torch.numel(weight),
                                                              p=[0.0, 1.0])
                                             .reshape(weight.shape))
                              )
                            for name, weight in weights.items()
                            }
            return mask

        def weight_generator(current_weights, mask, injecting_weights):
            # create a new model state
            result = {}
            # for each key
            for key, value in current_weights.items():
                # add the key
                result[key] = []
                # if False, replace initial value with trained value
                result[key] = current_weights[key].cuda().where(mask[key].cuda(), injecting_weights[key].cuda())
            return result


        def metropolis_train(position, inject_p, current_weights, injecting_weights):
            position = position
            inject_p = inject_p
            current_weights = current_weights
            injecting_weights = injecting_weights
            # identify the current layers to inject
            inject = ['bert.encoder.layer.%s.intermediate.dense.weight' % position,
                     'bert.encoder.layer.%s.output.dense.weight' % position]
            reject = ['attention']
            # create mask for the injection
            mask = mask_generator(inject_p=inject_p,
                                  inject=inject,
                                  reject=reject,
                                  weights=current_weights)
            # inject new weights
            injected_weights = weight_generator(current_weights=current_weights,
                                                    mask=mask,
                                                    injecting_weights=injecting_weights)
            # load the model with the new mix of weights
            self.model.load_state_dict(injected_weights)
            # train
            dev_acc, dev_precision, dev_recall, dev_f1, dev_loss = BertGLUEEvaluator(self.model, self.processor, self.args, self.logger).get_loss(type='dev')
            return dev_acc, injected_weights

        def metropolis(inject_p, duration, current_weights, injecting_weights):
            duration = duration
            inject_p = inject_p
            current_weights = current_weights
            injecting_weights = injecting_weights
            # collect metrics
            blocks_visited = np.zeros(duration)
            weights_changed = np.zeros(duration)
            metrics = np.zeros(duration)
            injection_rate = np.zeros(duration)
            # step 1: start randomly at layer 0-11
            current_position = np.random.randint(low=0, high=12)
            for i in range(duration):
                if inject_p >= 0.98:
                    break
                print('injecting this % of weights now', inject_p)
                injection_rate[i] = inject_p
                # record position
                blocks_visited[i] = current_position
                print('now on current:', current_position)
                # generate a proposal encoder block to visit
                proposal = current_position + np.random.choice([-1, 1])
                # loop around the encoder blocks
                if proposal < 0:
                    proposal = 11  # keep in the right layer ranges
                if proposal > 11:
                    proposal = 0
                print('proposal block:', proposal)
                # generate hypotheticals - evaluate on current
                dev_metric_current, current_injected_weights = metropolis_train(position=current_position,
                                                                                inject_p=inject_p,
                                                                                current_weights=current_weights,
                                                                                injecting_weights=injecting_weights)
                # generate hypotheticals - evaluate on proposal
                #dev_metric_proposal, proposal_injected_weights = metropolis_train(position=proposal,
                #                                                                  inject_p=inject_p,
                #                                                                  current_weights=current_weights,
                #                                                                  injecting_weights=injecting_weights)
                # report results
                print('current metric:', dev_metric_current)
                #print('proposal metric:', dev_metric_proposal)
                metrics[i] = dev_metric_current
                # if current block outperforms proposal block
                if dev_metric_current > 0.7:
                    current_weights = current_injected_weights
                    # record that a weight was altered at this layer
                    weights_changed[i] = 1
                    # otherwise proceed and draw comparions in next move
                if dev_metric_current < 0.7:
                    self.patience += 1
                    #if self.patience > 500:
                    #    break
                # move to new encoder block?
                if (current_position == 0) or (proposal == 0):
                    prob_move = (proposal+1) / (current_position+1)
                else:
                    prob_move = proposal / current_position
                if np.random.uniform() < prob_move:
                    # move
                    current_position = proposal
                duration -= 1
                if duration % 10 == 0:
                    inject_p += 0.001

            return blocks_visited, weights_changed, metrics, injection_rate

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
                #inject_p, duration, current_weights, trained_weights
                blocks_visited, weights_changed, metrics, injection_rate = metropolis(inject_p=self.inject_p,
                                            duration=self.duration,
                                            current_weights=self.trained_weights,
                                            injecting_weights=self.t0_weights)
                # trained_weights, t0_weights
                # print validation results
                #self.logger.info("Epoch {0: d}, Dev/Acc {1: 0.3f}, Dev/Pr. {2: 0.3f}, Dev/Re. {3: 0.3f}, Dev/F1 {4: 0.3f}, Dev/Loss {5: 0.3f}",
                #                 epoch+1, dev_acc, dev_precision, dev_recall, dev_f1, dev_loss)
                import pandas as pd
                result_df = pd.DataFrame({'blocks_visited': blocks_visited,
                                          'weights_changed': weights_changed,
                                          'metrics': metrics,
                                          'injection_rate': injection_rate})
                result_df.to_csv('metropolis_%s.csv' % self.args.model, index=False)

            return blocks_visited, weights_changed, metrics, injection_rate


        elif any([self.args.model == 'CoLA']):

            for epoch in trange(int(self.args.epochs), desc="Epoch"):
                # train
                self.train_epoch(train_dataloader)
                # get dev loss
                matthews, dev_loss = BertGLUEEvaluator(self.model, self.processor, self.args, self.logger).get_loss(type='dev')
                # print validation results
                self.logger.info("Epoch {0: d}, Dev/Matthews {1: 0.3f}, Dev/Loss {2: 0.3f}",
                                 epoch+1, matthews, dev_loss)

                self.epoch_loss.append(dev_loss)
                self.epoch_metric.append(matthews)
                self.epoch.append(epoch+1)
                self.epoch_freeze_p.append(self.freeze_p)

            return self.epoch_loss, self.epoch_metric, self.epoch, self.epoch_freeze_p


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
