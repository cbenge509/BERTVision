# packages
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import sys, os, random
sys.path.append("C:/BERTVision/code/torch")
from data.bert_processors.processors import *
from common.trainers.bert_search_trainer import BertSearchTrainer
from models.hypersearch.args import get_args
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler
from loguru import logger


def train_and_evaluate(lr, seed):
    # set default configuration in args.py
    args = get_args()
    # instantiate data set map; pulles the right processor / data for the task
    dataset_map = {
        'MSR': MSR,
        'CoLA': CoLA,
        'MNLI': MNLI,
        'QNLI': QNLI,
        'QQP': QQP,
        'RTE': RTE,
        'SST': SST,
        'STSB': STSB,
        'WNLI': WNLI
    }
    # tell the CLI user that they mistyped the data set
    if args.model not in dataset_map:
        raise ValueError('Unrecognized dataset')

    # set the location for saving the model
    save_path = os.path.join(args.save_path, args.checkpoint, args.model)
    os.makedirs(save_path, exist_ok=True)

    # set the location for saving the log
    log_path = os.path.join(args.log_path, args.checkpoint, args.model)
    os.makedirs(log_path, exist_ok=True)

    # initialize logging
    logger.add(log_path + '\\' + args.model + '.log', rotation="10 MB")
    logger.info(f"Training model {args.model} on this checkpoint: {args.checkpoint}")

    # set device to gpu/cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # detect number of gpus
    n_gpu = torch.cuda.device_count()
    # turn on autocast for fp16
    torch.cuda.amp.autocast(enabled=True)
    # set grad scaler
    scaler = GradScaler()

    # set seed for reproducibility
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    args.seed = seed

    # set some other training objects
    args.batch_size = args.batch_size
    args.device = device
    args.n_gpu = n_gpu
    args.lr = lr

    # make kwargs
    kwargs = args

    # set seed for multi-gpu
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # set data set processor
    processor = dataset_map[args.model]

    # shard the large datasets:
    if any([args.model == 'QQP',
            args.model == 'QNLI',
            args.model == 'MNLI',
            args.model == 'SST'
            ]):
        # turn on sharding
        train_processor = processor(type='train', transform=Tokenize_Transform(args, logger), shard=True, kwargs=kwargs)

    else:
        # create the usual processor
        train_processor = processor(type='train', transform=Tokenize_Transform(args, logger))

    # set training length
    num_train_optimization_steps = int(len(train_processor) / args.batch_size) * args.epochs

    # instantiate model and attach it to device
    model = BertForSequenceClassification.from_pretrained(args.checkpoint,
                                                          num_labels=args.num_labels).to(device)

    # print metrics
    logger.info(f"Device: {str(device).upper()}")
    logger.info(f"Number of GPUs: {n_gpu}")

    # for multi-GPU
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # set optimizer
    param_optimizer = list(model.named_parameters())

    # exclude these from regularization
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # give l2 regularization to any parameter that is not named after no_decay list
    # give no l2 regulariation to any bias parameter or layernorm bias/weight
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.l2},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    # set optimizer
    optimizer = AdamW(optimizer_grouped_parameters,
                              lr=args.lr,
                              correct_bias=False,
                              weight_decay=args.l2)

    # set linear scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=num_train_optimization_steps,
                                                num_warmup_steps=args.warmup_proportion * num_train_optimization_steps)

    # initialize the trainer
    trainer = BertSearchTrainer(model, optimizer, processor, scheduler, args, kwargs, scaler, logger)
    # begin training / shift to trainer class
    metric = trainer.train()
    return metric

# main fun.
if __name__ == '__main__':

    # training function
    def train_fn(params):
        seed = int(params['seed'])
        lr = params['lr']

        logger.info(f"Testing this learning rate: {lr} and this seed: {seed}")
        metric = train_and_evaluate(lr, seed)
        return {'loss': 1-metric, 'status': STATUS_OK}  # minimizing the fn, so 1-metric

    # search space
    search_space = {'seed': hp.randint('seed', 1000),
                    'lr': hp.uniform('lr', low=0e-5, high=3e-5)}

    # intialize hyperopt
    trials = Trials()

    argmin = fmin(
      fn=train_fn,
      space=search_space,
      algo=tpe.suggest,
      max_evals=100,
      trials=trials)

    # output argmin results
    logger.info(f"argmin results {argmin}")



#
