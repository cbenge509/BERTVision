# packages
from argparse import ArgumentParser
import sys
sys.path.append("C:/utils/utils")
from tools import AverageMeter, ProgressBar, format_time
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup, AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import time, datetime, json, h5py, pytreebank
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


# prepare torch data set
class SST5DataSet(torch.utils.data.Dataset):
    '''
    This prepares the official Stanford Sentiment Treebank
    (https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) for traininig
    in PyTorch. Download the files and place into the path folder.

    Parameters
    ----------
    is_finegrained : flag
        Whether or not we should do 5-label or 2-label sentiment analysis

    transform : optionable, callable flag
        Whether or not we need to tokenize transform the data

    Returns
    -------
    sample : dict
        A dictionary containing: (1) text, (2) labels,
        and index positions
    '''
    def __init__(self, is_finegrained=True, transform=None):
        self.path = 'C:\\utils\\data\\sst'
        # initialize flag; set false for binary classification
        self.is_finegrained = is_finegrained
        # binary categories: 1 and 3 only
        self.binary = [1, 3]
        # set sst path
        dataset = pytreebank.load_sst(self.path)

        # prase train test dev
        for category in ['train', 'test', 'dev']:
            with open(self.path + '\\' + 'sst_{}.txt'.format(category), 'w') as outfile:
                for item in dataset[category]:
                    outfile.write("{}\t{}\n".format(
                        item.to_labeled_lines()[0][0],
                        item.to_labeled_lines()[0][1]))

        # initialize train
        self.train = pd.read_csv(self.path + '\\' + 'sst_train.txt', sep='\t',
                                 header=None, names=['label', 'text'],
                                 encoding='latin-1')

        # filter if fine_grained
        if self.is_finegrained is False:
            self.train = self.train.loc[self.train['label'].isin(self.binary)].reset_index(drop=True)

            # map to binary then: 0, 1
            self.train['label'] = self.train['label'].map({1: 0, 3: 1})

        # initialize the transform if specified
        self.transform = transform

    # get len
    def __len__(self):
        return len(self.train)

    # pull a sample of data
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            print(idx)

        # return train, dev, test
        sample = {'text': self.train.text[idx],
                  'label': self.train.label[idx],
                  'idx': idx}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Tokenize_Transform():
    '''
    This function tokenize transforms the data organized in SST5DataSet().

    Parameters
    ----------
    sample : dict
        A dictionary containing: (1) text, (2) labels,
        and index positions

    Returns
    -------
    sample : dict
        A dictionary containing: (1) input tokens, (2) attention masks,
        (3) labels, and (4) data set index.
    '''
    def __init__(self, tokenizer):
        # instantiate the tokenizer
        self.tokenizer = tokenizer

    # retrieve sample and unpack it
    def __call__(self, sample):
        # transform text to input ids and attn masks
        encodings = self.tokenizer(
                            sample['text'],  # document to encode.
                            add_special_tokens=True,  # add '[CLS]' and '[SEP]'
                            max_length=64,  # set max length
                            truncation=True,  # truncate longer messages
                            padding='max_length',  # add padding
                            return_attention_mask=True,  # create attn. masks
                            return_tensors='pt'  # return pytorch tensors
                       )

        # package up encodings
        return {'input_ids': torch.as_tensor(encodings['input_ids'],
                                         dtype=torch.long),

                'attn_mask': torch.as_tensor(encodings['attention_mask'],
                                          dtype=torch.long),

                'label': torch.as_tensor(sample['label'],
                                          dtype=torch.long),

                'idx': torch.as_tensor(sample['idx'],
                                       dtype=torch.int)}


# train function
def train(model, dataloader, scaler, optimizer, device):
    pbar = ProgressBar(n_total=len(dataloader), desc='Training')
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    train_f1 = AverageMeter()
    count = 0
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        input_ids, attn_mask, label = (batch['input_ids'].to(device),
                                       batch['attn_mask'].to(device),
                                       batch['label'].to(device))
        optimizer.zero_grad()
        with autocast():
            out = model(input_ids=input_ids.squeeze(1),
                        attention_mask=attn_mask.squeeze(1),
                        labels=label)
        pred = out['logits'].argmax(dim=1, keepdim=True)
        correct = pred.eq(label.view_as(pred)).sum().item()
        f1 = f1_score(pred.cpu().numpy(), label.cpu().numpy(), average='weighted')
        scaler.scale(out['loss']).backward()
        scaler.step(optimizer)
        scaler.update()
        count += input_ids.size(0)
        pbar(step=batch_idx, info={'loss': out['loss'].item()})
        train_loss.update(out['loss'].item(), n=1)
        train_f1.update(f1, n=input_ids.size(0))
        train_acc.update(correct, n=1)
    return {'loss': train_loss.avg,
            'acc': train_acc.sum / count,
            'f1': train_f1.avg}


# prepare embedding extraction
def emit_embeddings(dataloader, train_dataset, model, device, args):
    # timing metrics
    t0 = time.time()
    batch_num = args.batch_size
    num_documents = len(train_dataset)
    # set a manipulable var to handle any batch size
    train_len = len(train_dataset)
    # check whether or not batch is divisible.
    if len(train_dataset) % args.batch_size != 0:
        remainder = len(train_dataset) % args.batch_size
        train_len = len(train_dataset) - remainder

    with h5py.File('C:\\w266\\data2\\h5py_embeds\\sst_bert_embeds.h5', 'w') as f:
        # create empty data set; [batch_sz, layers, tokens, features]
        dset = f.create_dataset('embeds', shape=(train_len, 13, args.max_seq_length, 768),
                                maxshape=(None, 13, args.max_seq_length, 768),
                                chunks=(args.batch_size, 13, args.max_seq_length, 768),
                                dtype=np.float32)

    with h5py.File('C:\\w266\\data2\\h5py_embeds\\sst_labels.h5', 'w') as l:
        # create empty data set; [batch_sz]
        label_dset = l.create_dataset('labels', shape=(train_len,),
                                      maxshape=(None,), chunks=(args.batch_size,),
                                      dtype=np.int64)

    print('Generating embeddings for all {:,} documents...'.format(len(train_dataset)))
    for step, batch in enumerate(dataloader):
        # send necessary items to GPU
        input_ids, attn_mask, label = (batch['input_ids'].to(device),
                                       batch['attn_mask'].to(device),
                                       batch['label'].to(device))

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
            # ['hidden_states'] is embeddings for all layers
            out = model(input_ids=input_ids.squeeze(1),
                        attention_mask=attn_mask.squeeze(1),
                        labels=label)

        # stack embeddings [layers, batch_sz, tokens, features]
        embeddings = torch.stack(out['hidden_states']).float()  # float32
        # swap the order to: [batch_sz, layers, tokens, features]
        # we need to do this to emit batches from h5 dataset later
        embeddings = embeddings.permute(1, 0, 2, 3).cpu().numpy()

        # add embeds to ds
        with h5py.File('C:\\w266\\data2\\h5py_embeds\\sst_bert_embeds.h5', 'a') as f:
            dset = f['embeds']
            # add chunk of rows
            start = step*args.batch_size
            # [batch_sz, layer, tokens, features]
            dset[start:start+args.batch_size, :, :, :] = embeddings[:, :, :, :]
            # Create attribute with last_index value
            dset.attrs['last_index'] = (step+1)*args.batch_size
            # check the integrity of the embeddings
            x = f['embeds'][start:start+args.batch_size, :, :, :]

        # add labels to ds
        with h5py.File('C:\\w266\\data2\\h5py_embeds\\sst_labels.h5', 'a') as l:
            label_dset = l['labels']
            # add chunk of rows
            start = step*args.batch_size
            # [batch_sz, layer, tokens, features]
            label_dset[start:start+args.batch_size] = label.cpu().numpy()
            # Create attribute with last_index value
            label_dset.attrs['last_index'] = (step+1)*args.batch_size

        batch_num += args.batch_size
        torch.cuda.empty_cache()

    # check data
    with h5py.File('C:\\w266\\data2\\h5py_embeds\\sst_bert_embeds.h5', 'r') as f:
        print('last embed batch entry', f['embeds'].attrs['last_index'])
        print('embed shape', f['embeds'].shape)
        print('last entry:', f['embeds'][-1, :, :, :])

    with h5py.File('C:\\w266\\data2\\h5py_embeds\\sst_labels.h5', 'r') as l:
        print('last embed batch entry', l['labels'].attrs['last_index'])
        print('embed shape', l['labels'].shape)
        print('last entry:', l['labels'][train_len-10: train_len])

    return None


def main():
    # training settings
    parser = ArgumentParser(description='SST5 Sentiment')
    parser.add_argument('--tokenizer', type=str,
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
                         help='learning rate (default: 1e-5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                         help='random seed (default: 1)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                         help='number of CPU cores (default: 4)')
    parser.add_argument('--num-labels', type=int, default=5, metavar='N',
                         help='number of labels to classify (default: 5)')
    parser.add_argument('--l2', type=float, default=1.0, metavar='LR',
                         help='l2 regularization weight (default: 1.0)')
    parser.add_argument('--max-seq-length', type=int, default=64, metavar='N',
                         help='max sequence length for encoding (default: 64)')
    args = parser.parse_args()

    # set seeds and determinism
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.amp.autocast(enabled=True)

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)

    # build df
    train_ds = SST5DataSet(is_finegrained=True, transform=Tokenize_Transform(tokenizer=tokenizer))

    # create training dataloader
    train_dataloader = DataLoader(train_ds,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  drop_last=True)

    # load the model
    model = BertForSequenceClassification.from_pretrained(args.model,
                                                          num_labels=args.num_labels).to(device)

    # create gradient scaler for mixed precision
    scaler = GradScaler()

    # create optimizer with L2
    optimizer = AdamW(model.parameters(),
                      lr=args.lr,
                      weight_decay=args.l2,
                      )

    # set epochs
    epochs = args.epochs

    # train
    best_loss = np.inf
    for epoch in range(1, epochs + 1):
        train_log = train(model, train_dataloader, scaler, optimizer, device)
        logs = dict(train_log)
        if logs['loss'] < best_loss:
            # torch save
            torch.save(model.state_dict(), 'C:\\w266\\data2\\checkpoints\\BERT-sst' + '_epoch_{}.pt'.format(epoch))
            best_loss = logs['loss']
        show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        print(show_info)

    # now proceed to emit embeddings
    model = BertForSequenceClassification.from_pretrained(args.model,
                                                          num_labels=args.num_labels,
                                                          output_hidden_states=True).to(device)
    # load weights from 1 epoch
    model.load_state_dict(torch.load('C:\\w266\\data2\\checkpoints\\BERT-sst_epoch_1.pt'))

    # export embeddings
    emit_embeddings(train_dataloader, train_ds, model, device, args)

# run program
if __name__ == '__main__':
    main()

#
