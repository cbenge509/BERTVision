# packages
import sys, os, random, datetime
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, rand
sys.path.append("C:/BERTVision/code/torch")
from data.bert_processors.processors import *
from common.trainers.bert_freeze_trainer import BertFreezeTrainer
from models.pfreezing.args import get_args
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler
from loguru import logger
import pickle as pkl


# main fun.
def train_and_evaluate(seed, inject, reject):
    # set default configuration in args.py
    args = get_args()
    # instantiate data set map; pulls the right processor / data for the task
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
    logger.add(log_path + '\\' + args.model + '.log', rotation="5000 MB")
    logger.info(f"Training model {args.model} on this checkpoint: {args.checkpoint}")

    # set device to gpu/cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # detect number of gpus
    n_gpu = torch.cuda.device_count()
    args.device = device
    args.n_gpu = n_gpu
    # turn on autocast for fp16
    torch.cuda.amp.autocast(enabled=True)
    # set grad scaler
    scaler = GradScaler()

    # don't freeze this select of weights
    #args.freeze = freeze

    # set the seed
    args.seed = seed
    args.inject = inject
    args.reject = reject

    # make kwargs
    kwargs = args

    # set seed for reproducibility
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set seed for multi-gpu
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

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
                                                num_warmup_steps=(args.warmup_proportion * num_train_optimization_steps))

    # initialize the trainer
    trainer = BertFreezeTrainer(model, optimizer, processor, scheduler, args, kwargs, scaler, logger)

    # begin training / shift to trainer class
    dev_loss, dev_metric, epoch, freeze_p = trainer.train()

    # return metrics
    return dev_loss, dev_metric, epoch, freeze_p

# execution
if __name__ == '__main__':

    # set default configuration in args.py
    args = get_args()

    # set the location for saving the model
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(args.pickle_path, args.checkpoint, args.model)
    os.makedirs(save_path, exist_ok=True)

    # training function
    def train_fn(params):
        # select params
        seed = int(params['seed'])
        inject = params['inject']
        reject = params['reject']
        #freeze = params['freeze']
        #freeze_p = params['freeze_p']
        # print info to user
        logger.info(f"""\n Starting trials with this seed: {seed} and this injection
                    {inject}""")
        # collect metrics
        dev_loss, dev_metric, epoch, freeze_p = train_and_evaluate(seed, inject, reject)
        # return metrics to trials
        return {'loss': 1, 'status': STATUS_OK, 'metric': dev_metric,
                'dev_loss': dev_loss, 'epoch': epoch, 'freeze_p': freeze_p}


    layers_0_3 = ['bert.encoder.layer.0.intermediate.dense.weight',
                 'bert.encoder.layer.0.intermediate.dense.bias',
                 'bert.encoder.layer.0.output.dense.weight',
                 'bert.encoder.layer.0.output.dense.bias',
                 'bert.encoder.layer.1.intermediate.dense.weight',
                 'bert.encoder.layer.1.intermediate.dense.bias',
                 'bert.encoder.layer.1.output.dense.weight',
                 'bert.encoder.layer.1.output.dense.bias',
                 'bert.encoder.layer.2.intermediate.dense.weight',
                 'bert.encoder.layer.2.intermediate.dense.bias',
                 'bert.encoder.layer.2.output.dense.weight',
                 'bert.encoder.layer.2.output.dense.bias',
                 'bert.encoder.layer.3.intermediate.dense.weight',
                 'bert.encoder.layer.3.intermediate.dense.bias',
                 'bert.encoder.layer.3.output.dense.weight',
                 'bert.encoder.layer.3.output.dense.bias']

    layers_6_9 = ['bert.encoder.layer.6.intermediate.dense.weight',
                 'bert.encoder.layer.6.intermediate.dense.bias',
                 'bert.encoder.layer.6.output.dense.weight',
                 'bert.encoder.layer.6.output.dense.bias',
                 'bert.encoder.layer.7.intermediate.dense.weight',
                 'bert.encoder.layer.7.intermediate.dense.bias',
                 'bert.encoder.layer.7.output.dense.weight',
                 'bert.encoder.layer.7.output.dense.bias',
                 'bert.encoder.layer.8.intermediate.dense.weight',
                 'bert.encoder.layer.8.intermediate.dense.bias',
                 'bert.encoder.layer.8.output.dense.weight',
                 'bert.encoder.layer.8.output.dense.bias',
                 'bert.encoder.layer.9.intermediate.dense.weight',
                 'bert.encoder.layer.9.intermediate.dense.bias',
                 'bert.encoder.layer.9.output.dense.weight',
                 'bert.encoder.layer.9.output.dense.bias']

    layers_8_11 = ['bert.encoder.layer.8.intermediate.dense.weight',
                 'bert.encoder.layer.8.intermediate.dense.bias',
                 'bert.encoder.layer.8.output.dense.weight',
                 'bert.encoder.layer.8.output.dense.bias',
                 'bert.encoder.layer.9.intermediate.dense.weight',
                 'bert.encoder.layer.9.intermediate.dense.bias',
                 'bert.encoder.layer.9.output.dense.weight',
                 'bert.encoder.layer.9.output.dense.bias',
                 'bert.encoder.layer.10.intermediate.dense.weight',
                 'bert.encoder.layer.10.intermediate.dense.bias',
                 'bert.encoder.layer.10.output.dense.weight',
                 'bert.encoder.layer.10.output.dense.bias',
                 'bert.encoder.layer.11.intermediate.dense.weight',
                 'bert.encoder.layer.11.intermediate.dense.bias',
                 'bert.encoder.layer.11.output.dense.weight',
                 'bert.encoder.layer.11.output.dense.bias']

    all_FFN = ['intermediate.dense', 'output.dense']
    pooler = ['pooler']
    classifier = ['classifier']
    reject = ['attention']

    # search space
    search_space = {'seed': hp.randint('seed', 1000),
                    'inject': hp.choice('freeze',
                                           [
                                           pooler,
                                           classifier,
                                           pooler + all_FFN,
                                           classifier + pooler,
                                           classifier + all_FFN
                                           ]
                                           ),
                    'reject': hp.choice('reject',
                                        [
                                         reject
                                        ])
                    }

    # intialize hyperopt
    trials = Trials()

    argmin = fmin(
      fn=train_fn,
      space=search_space,
      algo=rand.suggest,
      max_evals=args.n_trials,
      trials=trials)

    # results
    logger.info(f"Training complete!")

    # save the trials to save path and give it a timestamp filename
    pkl.dump(trials, open(save_path + '\\%s.pkl' % timestamp, "wb"))




#
