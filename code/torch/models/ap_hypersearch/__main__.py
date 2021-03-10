# packages
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import sys, os, random
sys.path.append("C:/BERTVision/code/torch")
from data.h5_processors.h5_processors import *
from utils.compress_utils import AdapterPooler, AP_GLUE
from common.trainers.H5_search_trainer import H5SearchTrainer
from models.ap_hypersearch.args import get_args
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizerFast
from loguru import logger
from torch.nn import MSELoss


def train_and_evaluate(lr, seed):
    # set default configuration in args.py
    args = get_args()

    # add some configs depending on checkpoint chosen
    if args.checkpoint == 'bert-base-uncased':
        args.n_layers = 13
        args.n_features = 768

    elif args.checkpoint == 'bert-large-uncased':
        args.n_layers = 25
        args.n_features = 1024

    # instantiate data set map; pulles the right processor / data for the task
    dataset_map = {
        'AP_MSR': MSRH5Processor,
        'AP_CoLA': COLAH5Processor,
        'AP_MNLI': MNLIH5Processor,
        'AP_QNLI': QNLIH5Processor,
        'AP_QQP': QQPH5Processor,
        'AP_RTE': RTEH5Processor,
        'AP_SST': SSTH5Processor,
        'AP_STSB': STSBH5Processor,
        'AP_WNLI': WNLIH5Processor
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

    # set seed for multi-gpu
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # set data set processor
    processor = dataset_map[args.model]

    # set some other training objects
    args.batch_size = args.batch_size
    args.device = device
    args.n_gpu = n_gpu
    args.lr = lr

    # instantiate model and attach it to device
    model = AP_GLUE(n_layers=args.n_layers, n_batch_sz=args.batch_size, n_tokens=args.max_seq_length, n_features=args.n_features, n_labels=args.num_labels).to(device)

    # set loss
    if args.num_labels == 1:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # print metrics
    logger.info(f"Device: {str(device).upper()}")
    logger.info(f"Number of GPUs: {n_gpu}")

    # for multi-GPU
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # initialize the trainer
    trainer = H5SearchTrainer(model, processor, criterion, args, scaler, logger)
    # begin training / shift to trainer class
    metric = trainer.train()

    return metric

# main fun.
if __name__ == '__main__':
    args = get_args()

    # training function
    def train_fn(params):
        seed = int(params['seed'])
        lr = params['lr']

        logger.info(f"Testing this learning rate: {lr} and this seed: {seed}")
        metric = train_and_evaluate(lr, seed)
        return {'loss': 1-metric, 'status': STATUS_OK}  # minimizing the fn, so 1-metric

    # search space
    search_space = {'seed': hp.randint('seed', 1000),
                    'lr': hp.uniform('lr', low=args.lr_low, high=args.lr_high)}

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
