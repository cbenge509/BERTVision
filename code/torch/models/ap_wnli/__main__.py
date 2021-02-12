# packages
import sys, os, random
sys.path.append("C:/BERTVision/code/torch")
from data.h5_processors.h5_processors import WNLIH5Processor
from utils.compress_utils import AdapterPooler, AP_GLUE
from common.trainers.H5_glue_trainer import H5_GLUE_Trainer
from models.ap_wnli.args import get_args
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizerFast
from loguru import logger

# main fun.
if __name__ == '__main__':
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
        'AP_WNLI': WNLIH5Processor
    }

    # tell the CLI user that they mistyped the data set
    args.dataset = 'AP_WNLI'
    if args.dataset not in dataset_map:
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set seed for multi-gpu
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # instantiate model and attach it to device
    model = AP_GLUE(n_layers=args.n_layers, n_batch_sz=args.batch_size, n_tokens=args.max_seq_length, n_features=args.n_features, n_labels=args.num_labels).to(device)
    # set data set processor
    processor = dataset_map[args.dataset]
    # use it to create the train set
    train_processor = processor(type='train', args=args)
    # set loss
    criterion = nn.CrossEntropyLoss()

    # set some other training objects
    args.batch_size = args.batch_size
    args.device = device
    args.n_gpu = n_gpu

    # find number of optim. steps
    num_train_optimization_steps = int(len(train_processor) / args.batch_size) * args.epochs

    # print metrics
    logger.info(f"Device: {str(device).upper()}")
    logger.info(f"Number of GPUs: {n_gpu}")

    # for multi-GPU
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # set optimizer
    optimizer = AdamW(model.parameters(),
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
