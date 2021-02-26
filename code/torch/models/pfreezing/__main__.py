# packages
import sys, os, random
sys.path.append("C:/BERTVision/code/torch")
from data.bert_processors.processors import *
from common.trainers.bert_freeze_trainer import BertFreezeTrainer
from models.bert_glue.args import get_args
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler
from loguru import logger


# main fun.
def train_and_evaluate(seed, freeze):
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # set seed for multi-gpu
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # set data set processor
    processor = dataset_map[args.model]

    # use it to create the train set
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

    # paramter freezing: exclude these from freezing
    no_freeze = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'embeddings']

    # locate randomly selected weights
    locked_masks = {
                    name: torch.tensor(np.random.choice([False, True],
                                                  size=torch.numel(weight),
                                                  p=[(1-freeze), freeze]).reshape(weight.shape))
                    for name, weight in model.named_parameters()
                    if not any(weight in name for weight in no_freeze)
                    }

    # set linear scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=num_train_optimization_steps,
                                                num_warmup_steps=(args.warmup_proportion * num_train_optimization_steps))

    # initialize the trainer
    trainer = BertFreezeTrainer(model, optimizer, processor, scheduler, args, scaler, logger, locked_masks)

    # begin training / shift to trainer class
    metric = trainer.train()

    return freeze, seed, metric

# execution
if __name__ == '__main__':

    # set default configuration in args.py
    args = get_args()

    # trial search space
    search_space = np.linspace(start=0, stop=1, num=50)

    # container storage
    m_trial, m_freeze, m_seed, m_metric = [], [], [], []

    for trial, freeze_p in enumerate(search_space):
        logger.info(f"Now freezing this proportion of parameters: {freeze_p}")
        seed = np.random.randint(0, 1000)
        np.random.seed(seed)
        # experiment
        freeze, seed, metric = train_and_evaluate(seed=seed, freeze=freeze_p)

        # output results
        if any([args.model == 'SST',
                args.model == 'MSR',
                args.model == 'RTE',
                args.model == 'QNLI',
                args.model == 'QQP',
                args.model == 'WNLI',
                args.model == 'MNLI'
                ]):
            logger.info("Epoch {0: d}, Freeze Prop. {1: 0.4f}, Seed. {2: d}, Dev/Acc. {3: 0.3f}",
                 trial+1, freeze, seed, metric)

        elif any([args.model == 'STSB']):
            logger.info("Epoch {0: d}, Freeze Prop. {1: 0.4f}, Seed. {2: d}, Dev/Pearson. {3: 0.3f}",
                     trial+1, freeze, seed, metric)

        elif any([args.model == 'CoLA']):
            logger.info("Epoch {0: d}, Freeze Prop. {1: 0.4f}, Seed. {2: d}, Dev/Mathews. {3: 0.3f}",
                     trial+1, freeze, seed, metric)

        # store some results
        m_trial.append(trial)
        m_freeze.append(freeze)
        m_seed.append(seed)
        m_metric.append(metric)

    # store results to df
    df = pd.DataFrame({'trial': m_trial, 'freeze': m_freeze, 'seed': m_seed,
                       'metric': m_metric})
    # save to disk
    df.to_csv('freeze_bert.csv', index=False)

#

#
