# packages
import sys, os, random
sys.path.append("C:/BERTVision/code/torch")
from data.h5_processors.cola_H5_processor import COLAH5Processor
from utils.compress_utils import AdapterPooler, SST_AP
from common.trainers.H5_sst_trainer import H5_SST_Trainer
from models.ap_cola.args import get_args
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizerFast


# main fun.
if __name__ == '__main__':
    # set default configuration in args.py
    args = get_args()

    # instantiate data set map; pulles the right processor / data for the task
    dataset_map = {
        'COLAH5': COLAH5Processor
    }

    # tell the CLI user that they mistyped the data set
    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')

    # set the location for saving the model
    save_path = os.path.join(args.save_path, dataset_map[args.dataset].NAME, args.model_name)
    os.makedirs(save_path, exist_ok=True)

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
    model = SST_AP(n_layers=13, n_batch_sz=args.batch_size, n_tokens=args.max_seq_length, n_features=768, n_labels=args.num_labels).to(device)
    # set data set processor
    processor = dataset_map[args.dataset]
    # use it to create the train set
    train_processor = processor(type='train')
    # set loss
    criterion = nn.CrossEntropyLoss()

    # set some other training objects
    args.batch_size = args.batch_size
    args.device = device
    args.n_gpu = n_gpu
    num_train_optimization_steps = int(len(train_processor) / args.batch_size) * args.epochs

    # print metrics
    print('Device:', str(device).upper())
    print('Number of GPUs:', n_gpu)

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
    trainer = H5_SST_Trainer(model, criterion, optimizer, processor, scheduler, args, scaler)
    # begin training / shift to trainer class
    trainer.train()
    # load the checkpoint
    model = torch.load(trainer.snapshot_path)




#
