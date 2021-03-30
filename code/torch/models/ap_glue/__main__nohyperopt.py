# packages
import sys, os, random
sys.path.append("C:/BERTVision/code/torch")
from data.h5_processors.h5_processors import *
from utils.compress_utils import AdapterPooler, AP_GLUE
from common.trainers.H5_glue_trainer import H5_GLUE_Trainer
from models.ap_glue.args import get_args
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizerFast
from loguru import logger
from torch.nn import MSELoss

# main fun.
if __name__ == '__main__':
    # set default configuration in args.py
    args = get_args()
    if args.adapter_dim == 0:
        args.adapter_dim = args.max_seq_length

    # add some configs depending on checkpoint chosen
    if args.checkpoint == 'bert-base-uncased':
        args.n_layers = 13
        args.n_features = 768

    elif args.checkpoint == 'bert-large-uncased':
        args.n_layers = 25
        args.n_features = 1024

    # instantiate data set map; pulls the right processor / data for the task
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
    args.device = device
    args.n_gpu = n_gpu
    # turn on autocast for fp16
    torch.cuda.amp.autocast(enabled=True)
    # set grad scaler
    scaler = GradScaler()

    # set seed for reproducibility
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set seed for multi-gpu
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # instantiate model and attach it to device
    model = AP_GLUE(n_layers=args.n_layers, n_batch_sz=args.batch_size, n_tokens=args.max_seq_length, n_features=args.n_features, n_labels=args.num_labels, adapter_dim = args.adapter_dim).to(device)
    # set data set processor
    processor = dataset_map[args.model]
    # use it to create the train set
    train_processor = processor(type='train', args=args)
    # set loss
    if args.num_labels == 1:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # find number of optim. steps
    num_train_optimization_steps = int(len(train_processor) / args.batch_size) * args.epochs

    # print metrics
    logger.info(f"Device: {str(device).upper()}")
    logger.info(f"Number of GPUs: {n_gpu}")

    # for multi-GPU
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

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
    optimizer = AdamW(optimizer_grouped_parameters,
                              lr=args.lr,
                              correct_bias=False,
                              weight_decay=args.l2)

    # set linear scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=num_train_optimization_steps,
                                                num_warmup_steps=args.warmup_proportion * num_train_optimization_steps)

    # initialize the trainer
    trainer = H5_GLUE_Trainer(model, criterion, optimizer, processor, scheduler, args, scaler, logger)
    # begin training / shift to trainer class
    trainer.train()
    # load the checkpoint
    model = torch.load(trainer.snapshot_path)


#
