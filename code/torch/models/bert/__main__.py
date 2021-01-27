# packages
import sys, os, random
sys.path.append("C:/BERTVision")
from data.bert_processors.squad_processor import SQuADProcessor
from common.trainers.bert_trainer import BertTrainer
from models.bert.args import get_args
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler


# main fun.
if __name__ == '__main__':
    # set default configuration in args.py
    args = get_args()

    # instantiate data set map; pulles the right processor / data for the task
    dataset_map = {
        'SQuAD': SQuADProcessor
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
    model = BertForQuestionAnswering.from_pretrained(args.model_name).to(device)
    # set tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    # set data set processor
    processor = dataset_map[args.dataset]
    # use it to create the train set
    train_processor = processor(type='train')
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
    trainer = BertTrainer(model, optimizer, processor, scheduler, args, scaler)
    # begin training / shift to trainer class
    trainer.train()
    # load the checkpoint
    model = torch.load(trainer.snapshot_path)




#