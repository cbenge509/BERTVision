# packages
import os, sys, datetime
sys.path.append("C:/BERTVision/code/torch")
from common.evaluators.H5_sst_evaluator import H5_SST_Evaluator
from utils.collate import collate_H5_SST
from torch.cuda.amp import autocast
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tqdm.notebook import trange



class H5_SST_Trainer(object):
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
    def __init__(self, model, criterion, optimizer, processor, scheduler, args, scaler):
        # pull in objects
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.processor = processor
        self.scheduler = scheduler
        self.scaler = scaler
        # specify training data set
        self.train_examples = processor(type='train')
        # create a timestamp for the checkpoints
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # create a location to save the files
        self.snapshot_path = os.path.join(self.args.save_path, self.processor.NAME, self.args.model_name, '%s.pt' % timestamp)
        # determine the number of optimization steps
        self.num_train_optimization_steps = int(
            len(self.train_examples) / args.batch_size) * args.epochs

        # set log info and template
        self.log_header = 'Epoch Iteration Progress   Dev/Exact  Dev/F1   Dev/Loss'
        self.log_template = ' '.join('{:>5.0f},{:>9.0f},{:>6.0f}/{:<5.0f} {:>6.4f},{:>8.4f},{:8.4f}'.split(','))

        # create placeholders for model metrics and early stopping if desired
        self.iterations, self.nb_tr_steps, self.tr_loss = 0, 0, 0
        self.best_dev_f1, self.unimproved_iters = 0, 0
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

            # get loss for start and ending positions
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
            #print('\n', 'batch loss', loss.item())
        # print end of trainig results
        print('\n', 'train loss', self.tr_loss / self.nb_tr_steps)

    def train(self):
        '''
        This function handles the entirety of the training, dev, and scoring.
        '''
        # tell the user general metrics
        print("Number of examples: ", len(self.train_examples))
        print("Batch size:", self.args.batch_size)
        print("Number of optimization steps:", self.num_train_optimization_steps)

        # instantiate dataloader
        train_dataloader = DataLoader(self.train_examples,
                                      batch_size=self.args.batch_size,
                                      shuffle=True,
                                      num_workers=self.args.num_workers,
                                      drop_last=True,  # need to swap to False; find error
                                      collate_fn=collate_H5_SST)
        # for each epoch
        for epoch in trange(int(self.args.epochs), desc="Epoch"):
            # train
            self.train_epoch(self.criterion, train_dataloader)
            # get dev loss
            dev_acc, dev_precision, dev_recall, dev_f1, dev_loss = H5_SST_Evaluator(self.model, self.criterion, self.processor, self.args).get_loss()

            # print validation results
            tqdm.write(self.log_header)
            tqdm.write(self.log_template.format(epoch + 1, self.iterations, epoch + 1, self.args.epochs,
                                                dev_acc, dev_precision, dev_recall, dev_f1, dev_loss))

            # update validation results
            if dev_f1 > self.best_dev_f1:
                self.unimproved_iters = 0
                self.best_dev_f1 = dev_f1
                torch.save(self.model, self.snapshot_path)

            else:
                # stop training with early stopping
                self.unimproved_iters += 1
                if self.unimproved_iters >= self.args.patience:
                    self.early_stop = True
                    tqdm.write("Early Stopping. Epoch: {}, Best Dev F1: {}".format(epoch, self.best_dev_f1))
                    break

#
