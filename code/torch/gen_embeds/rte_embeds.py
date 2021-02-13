# packages
from argparse import ArgumentParser
import sys, os
sys.path.append("C:/BERTVision/code/torch")
from data.bert_processors.processors import RTE, Tokenize_Transform
from utils.tools import AverageMeter, ProgressBar, format_time
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup, AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.cuda.amp import autocast, GradScaler
import time, datetime, h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import csv
from loguru import logger

# train function
def train(model, dataloader, scaler, optimizer, scheduler, device, args):
    pbar = ProgressBar(n_total=len(dataloader), desc='Training')
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    train_f1 = AverageMeter()
    count = 0
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        input_ids, attn_mask, token_type_ids, label, idx = (batch['input_ids'].to(device),
                                       batch['attention_mask'].to(device),
                                       batch['token_type_ids'].to(device),
                                       batch['labels'].to(device),
                                       batch['idx'].to(device))
        optimizer.zero_grad()
        with autocast():
            out = model(input_ids=input_ids.squeeze(1),
                        attention_mask=attn_mask.squeeze(1),
                        token_type_ids=token_type_ids.squeeze(1),
                        labels=label)

        if args.num_labels > 1:
            pred = out['logits'].argmax(dim=1, keepdim=True)
            correct = pred.eq(label.view_as(pred)).sum().item()
            f1 = f1_score(pred.cpu().numpy(), label.cpu().numpy(), average='weighted')
            train_f1.update(f1, n=input_ids.size(0))
            train_acc.update(correct, n=1)
        else:
            pred = out['logits']

        scaler.scale(out['loss']).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        count += input_ids.size(0)
        pbar(step=batch_idx, info={'loss': train_loss.avg})
        train_loss.update(out['loss'].item(), n=1)
    return {'loss': train_loss.avg,
            'acc': train_acc.sum / count,
            'f1': train_f1.avg}


# prepare embedding extraction
def emit_train_embeddings(dataloader, train_dataset, model, device, args):
    # timing metrics
    t0 = time.time()
    batch_num = args.embed_batch_size
    num_documents = len(train_dataset)

    # set file location and layer / feature information
    if args.checkpoint == 'bert-base-uncased':
        save_location = 'C:\\w266\\data\\h5py_embeds\\'
        args.n_layers = 13
        args.n_features = 768
    else:
        save_location = 'C:\\w266\\data\\h5py_embeds\\bert_large\\'
        args.n_layers = 25
        args.n_features = 1024
    # create the dirs
    os.makedirs(save_location, exist_ok=True)

    with h5py.File(save_location + 'rte_bert_embeds.h5', 'w') as f:
        # create empty data set; [batch_sz, layers, tokens, features]
        dset = f.create_dataset('embeds', shape=(num_documents, args.n_layers, args.max_seq_length, args.n_features),
                                maxshape=(None, args.n_layers, args.max_seq_length, args.n_features),
                                chunks=(args.embed_batch_size, args.n_layers, args.max_seq_length, args.n_features),
                                dtype=np.float32)

    with h5py.File(save_location + 'rte_labels.h5', 'w') as l:
        # create empty data set; [batch_sz]
        label_dset = l.create_dataset('labels', shape=(num_documents,),
                                      maxshape=(None,), chunks=(args.embed_batch_size,),
                                      dtype=np.int64)

    with h5py.File(save_location + 'rte_idx.h5', 'w') as i:
        # create empty data set; [batch_sz]
        idx_dset = i.create_dataset('idx', shape=(num_documents,),
                                      maxshape=(None,), chunks=(args.embed_batch_size,),
                                      dtype=np.int64)

    print('Generating embeddings for all {:,} documents...'.format(len(train_dataset)))
    for step, batch in enumerate(dataloader):
        # send necessary items to GPU
        input_ids, attn_mask, token_type_ids, label, idx = (batch['input_ids'].to(device),
                                       batch['attention_mask'].to(device),
                                       batch['token_type_ids'].to(device),
                                       batch['labels'].to(device),
                                       batch['idx'].to(device))

        if step % 20 == 0 and not batch_num == 0:
            # calc elapsed time
            elapsed = format_time(time.time() - t0)
            # calc time remaining
            rows_per_sec = (time.time() - t0) / batch_num
            remaining_sec = rows_per_sec * (num_documents - batch_num)
            remaining = format_time(remaining_sec)
            # report progress
            print('Documents {:>7,} of {:>7,}. Elapsed: {:}. Remaining: {:}'.format(batch_num, num_documents, elapsed, remaining))

        # get embeddings with no gradient calcs
        with torch.no_grad():

            out = model(input_ids=input_ids.squeeze(1),
                        attention_mask=attn_mask.squeeze(1),
                        token_type_ids=token_type_ids.squeeze(1),
                        labels=label)

        # ['hidden_states'] is embeddings for all layers
        # stack embeddings [layers, batch_sz, tokens, features]
        embeddings = torch.stack(out['hidden_states']).float()  # float32
        # swap the order to: [batch_sz, layers, tokens, features]
        # we need to do this to emit batches from h5 dataset later
        embeddings = embeddings.permute(1, 0, 2, 3).cpu().numpy()

        # add embeds to ds
        with h5py.File(save_location + 'rte_bert_embeds.h5', 'a') as f:
            # initialize dset
            dset = f['embeds']
            # counter to add chunk of rows
            start = step*args.embed_batch_size
            # add to the dset              [batch_sz, layer, tokens, features]
            dset[start:start+args.embed_batch_size, :, :, :] = embeddings[:, :, :, :]
            # create attribute with last_index value
            dset.attrs['last_index'] = (step+1)*args.embed_batch_size

        # add labels to ds
        with h5py.File(save_location + 'rte_labels.h5', 'a') as l:
            # initialize dset
            label_dset = l['labels']
            # counter to add chunk of rows
            start = step*args.embed_batch_size
            # add to the dset              [batch_sz, ]
            label_dset[start:start+args.embed_batch_size] = label.cpu().numpy()
            # create attribute with last_index value
            label_dset.attrs['last_index'] = (step+1)*args.embed_batch_size

        # add idx to ds
        with h5py.File(save_location + 'rte_idx.h5', 'a') as i:
            # initialize dset
            idx_dset = i['idx']
            # counter to add chunk of rows
            start = step*args.embed_batch_size
            # [batch_sz, ]
            idx_dset[start:start+args.embed_batch_size] = idx.cpu().numpy()
            # create attribute with last_index value
            idx_dset.attrs['last_index'] = (step+1)*args.embed_batch_size

        batch_num += args.embed_batch_size
        torch.cuda.empty_cache()

    # check data
    with h5py.File(save_location + 'rte_bert_embeds.h5', 'r') as f:
        print('last embed batch entry', f['embeds'].attrs['last_index'])
        print('embed shape', f['embeds'].shape)
        print('last entry:', f['embeds'][-1, :, :, :])

    with h5py.File(save_location + 'rte_labels.h5', 'r') as l:
        print('last embed batch entry', l['labels'].attrs['last_index'])
        print('embed shape', l['labels'].shape)
        print('last entry:', l['labels'][len(train_dataset)-10: len(train_dataset)])

    return None


# prepare embedding extraction
def emit_dev_embeddings(dataloader, train_dataset, model, device, args):
    # timing metrics
    t0 = time.time()
    batch_num = args.embed_batch_size
    num_documents = len(train_dataset)

    # set file location and layer / feature information
    if args.checkpoint == 'bert-base-uncased':
        save_location = 'C:\\w266\\data\\h5py_embeds\\'
        args.n_layers = 13
        args.n_features = 768
    else:
        save_location = 'C:\\w266\\data\\h5py_embeds\\bert_large\\'
        args.n_layers = 25
        args.n_features = 1024
    # create the dirs
    os.makedirs(save_location, exist_ok=True)

    with h5py.File(save_location + 'rte_dev_bert_embeds.h5', 'w') as f:
        # create empty data set; [batch_sz, layers, tokens, features]
        dset = f.create_dataset('embeds', shape=(num_documents, args.n_layers, args.max_seq_length, args.n_features),
                                maxshape=(None, args.n_layers, args.max_seq_length, args.n_features),
                                chunks=(args.embed_batch_size, args.n_layers, args.max_seq_length, args.n_features),
                                dtype=np.float32)

    with h5py.File(save_location + 'rte_dev_labels.h5', 'w') as l:
        # create empty data set; [batch_sz]
        label_dset = l.create_dataset('labels', shape=(num_documents,),
                                      maxshape=(None,), chunks=(args.embed_batch_size,),
                                      dtype=np.int64)

    with h5py.File(save_location + 'rte_dev_idx.h5', 'w') as i:
        # create empty data set; [batch_sz]
        idx_dset = i.create_dataset('idx', shape=(num_documents,),
                                      maxshape=(None,), chunks=(args.embed_batch_size,),
                                      dtype=np.int64)

    print('Generating embeddings for all {:,} documents...'.format(num_documents))
    for step, batch in enumerate(dataloader):
        # send necessary items to GPU
        input_ids, attn_mask, token_type_ids, label, idx = (batch['input_ids'].to(device),
                                       batch['attention_mask'].to(device),
                                       batch['token_type_ids'].to(device),
                                       batch['labels'].to(device),
                                       batch['idx'].to(device))

        if step % 20 == 0 and not batch_num == 0:
            # calc elapsed time
            elapsed = format_time(time.time() - t0)
            # calc time remaining
            rows_per_sec = (time.time() - t0) / batch_num
            remaining_sec = rows_per_sec * (num_documents - batch_num)
            remaining = format_time(remaining_sec)
            # report progress
            print('Documents {:>7,} of {:>7,}. Elapsed: {:}. Remaining: {:}'.format(batch_num, num_documents, elapsed, remaining))

        # get embeddings with no gradient calcs
        with torch.no_grad():

            out = model(input_ids=input_ids.squeeze(1),
                        attention_mask=attn_mask.squeeze(1),
                        token_type_ids=token_type_ids.squeeze(1),
                        labels=label)

        # ['hidden_states'] is embeddings for all layers
        # stack embeddings [layers, batch_sz, tokens, features]
        embeddings = torch.stack(out['hidden_states']).float()  # float32
        # swap the order to: [batch_sz, layers, tokens, features]
        # we need to do this to emit batches from h5 dataset later
        embeddings = embeddings.permute(1, 0, 2, 3).cpu().numpy()

        # add embeds to ds
        with h5py.File(save_location + 'rte_dev_bert_embeds.h5', 'a') as f:
            # initialize dset
            dset = f['embeds']
            # counter to add chunk of rows
            start = step*args.embed_batch_size
            # insert into dset,             [batch_sz, layer, tokens, features]
            dset[start:start+args.embed_batch_size, :, :, :] = embeddings[:, :, :, :]
            # create attribute with last_index value
            dset.attrs['last_index'] = (step+1)*args.embed_batch_size

        # add labels to ds
        with h5py.File(save_location + 'rte_dev_labels.h5', 'a') as l:
            # initialize dset
            label_dset = l['labels']
            # counter to add chunk of rows
            start = step*args.embed_batch_size
            #  insert into dset,                            [batch_sz,]
            label_dset[start:start+args.embed_batch_size] = label.cpu().numpy()
            # create attribute with last_index value
            label_dset.attrs['last_index'] = (step+1)*args.embed_batch_size

        # add idx to ds
        with h5py.File(save_location + 'rte_dev_idx.h5', 'a') as i:
            # initialize dset
            idx_dset = i['idx']
            # counter to add chunk of rows
            start = step*args.embed_batch_size
            # [batch_sz, layer, tokens, features]
            idx_dset[start:start+args.embed_batch_size] = idx.cpu().numpy()
            # create attribute with last_index value
            idx_dset.attrs['last_index'] = (step+1)*args.embed_batch_size

        batch_num += args.embed_batch_size
        torch.cuda.empty_cache()

    # check data
    with h5py.File(save_location + 'rte_dev_bert_embeds.h5', 'r') as f:
        print('last embed batch entry', f['embeds'].attrs['last_index'])
        print('embed shape', f['embeds'].shape)
        print('last entry:', f['embeds'][-1, :, :, :])

    with h5py.File(save_location + 'rte_dev_labels.h5', 'r') as l:
        print('last embed batch entry', l['labels'].attrs['last_index'])
        print('embed shape', l['labels'].shape)
        print('last entry:', l['labels'][len(train_dataset)-10: len(train_dataset)])

    return None


def main():
    # training settings
    def get_args():
        parser = ArgumentParser(description='RTE - Recognizing Textual Entailment')
        parser.add_argument('--name', type=str,
                            default='RTE', metavar='S',
                            help="Model name")
        parser.add_argument('--checkpoint', type=str,
                            default='bert-base-uncased', metavar='S',
                            help="e.g., bert-base-uncased, etc")
        parser.add_argument('--model', type=str,
                            default='bert-base-uncased', metavar='S',
                            help="e.g., bert-base-uncased, etc")
        parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                             help='input batch size for training (default: 16)')
        parser.add_argument('--epochs', type=int, default=1, metavar='N',
                             help='number of epochs to train (default: 1)')
        parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                             help='learning rate (default: 3e-5)')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                             help='random seed (default: 1)')
        parser.add_argument('--num-workers', type=int, default=0, metavar='N',
                             help='number of CPU cores (default: 0)')
        parser.add_argument('--num-labels', type=int, default=2, metavar='N',
                             help='number of labels to classify (default: 2)')
        parser.add_argument('--l2', type=float, default=0.01, metavar='LR',
                             help='l2 regularization weight (default: 0.01)')
        parser.add_argument('--max-seq-length', type=int, default=219, metavar='N',
                             help='max sequence length for encoding (default: 219)')
        parser.add_argument('--warmup-proportion', type=int, default=0.1, metavar='N',
                             help='Warmup proportion (default: 0.1)')
        parser.add_argument('--embed-batch-size', type=int, default=1, metavar='N',
                             help='Embedding batch size emission; (default: 1)')
        args = parser.parse_args()
        return args

    args = get_args()

    # set seeds and determinism
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.amp.autocast(enabled=True)

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # build ds
    train_ds = RTE(type='train', transform=Tokenize_Transform(args, logger))

    # build ds
    dev_ds = RTE(type='dev', transform=Tokenize_Transform(args, logger))

    # create training dataloader
    train_dataloader = DataLoader(train_ds,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  drop_last=False)

    # create embed dataloader
    train_embed_dataloader = DataLoader(train_ds,
                                batch_size=args.embed_batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                drop_last=False)

    # create embed dataloader
    dev_embed_dataloader = DataLoader(dev_ds,
                                batch_size=args.embed_batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                drop_last=False)

    # load the model
    model = BertForSequenceClassification.from_pretrained(args.checkpoint,
                                                          num_labels=args.num_labels).to(device)

    # create gradient scaler for mixed precision
    scaler = GradScaler()

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

    num_train_optimization_steps = int(len(train_ds) / args.batch_size) * args.epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=num_train_optimization_steps,
                                                num_warmup_steps=args.warmup_proportion * num_train_optimization_steps)

    # set epochs
    epochs = args.epochs

    # set location and make if necessary
    if args.checkpoint == 'bert-base-uncased':
        checkpoint_location = 'C:\\w266\\data\\embed_checkpoints\\'
    elif args.checkpoint == 'bert-large-uncased':
        checkpoint_location = 'C:\\w266\\data\\embed_checkpoints\\bert_large\\'
    os.makedirs(checkpoint_location, exist_ok=True)

    # train
    best_loss = np.inf
    for epoch in range(1, epochs + 1):
        train_log = train(model, train_dataloader, scaler, optimizer, scheduler, device, args)
        logs = dict(train_log)
        if logs['loss'] < best_loss:
            # torch save
            torch.save(model.state_dict(), checkpoint_location + args.name + '_epoch_{}.pt'.format(epoch))
            best_loss = logs['loss']
        show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        print(show_info)

    # now proceed to emit embeddings
    model = BertForSequenceClassification.from_pretrained(args.checkpoint,
                                                          num_labels=args.num_labels,
                                                          output_hidden_states=True).to(device)
    # load weights from 1 epoch
    model.load_state_dict(torch.load(checkpoint_location + args.name + '_epoch_1.pt'))

    # export embeddings
    emit_train_embeddings(train_embed_dataloader, train_ds, model, device, args)
    emit_dev_embeddings(dev_embed_dataloader, dev_ds, model, device, args)

# run program
if __name__ == '__main__':
    main()

#
