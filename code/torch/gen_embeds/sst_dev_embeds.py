# packages
from argparse import ArgumentParser
import sys
sys.path.append("C:/BERTVision/code/torch")
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

class SSTProcessor(torch.utils.data.Dataset):
    '''
    This prepares the official Stanford Sentiment Treebank
    (https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf) for traininig
    in PyTorch. Download the files and place into the path folder.

    Parameters
    ----------
    is_multilabel : flag
        Whether or not we should do 5-label or 2-label sentiment analysis

    transform : optionable, callable flag
        Whether or not we need to tokenize transform the data

    Returns
    -------
    sample : dict
        A dictionary containing: (1) text, (2) labels,
        and index positions
    '''

    NAME = 'SST'

    def __init__(self, type, transform=None):
        # set path for data
        self.path = 'C:\\BERTVision\\code\\torch\\data\\data_sets\\sst\\trees'
        # initialize flag; set false for binary classification
        self.is_multilabel = False
        # flag for multilabel
        if self.is_multilabel == True:
            # then set the labels
            self.num_labels = 5
        else:
            self.num_labels = 2
        # binary categories: 1 and 3 only
        self.binary = [1, 3]
        # set type; train or dev?
        self.type = type

        if self.type == 'train':
            # initialize train
            self.train = pd.read_csv(self.path + '\\' + 'sst_train.txt', sep='\t',
                                     header=None, names=['label', 'text'],
                                     encoding='latin-1')

            # filter if fine_grained
            if self.is_multilabel is False:
                self.train = self.train.loc[self.train['label'].isin(self.binary)].reset_index(drop=True)
                # map to binary then: 0, 1
                self.train['label'] = self.train['label'].map({1: 0, 3: 1})

        if self.type == 'dev':
            # initialize dev
            self.dev = pd.read_csv(self.path + '\\' + 'sst_dev.txt', sep='\t',
                                     header=None, names=['label', 'text'],
                                     encoding='latin-1')

            # filter if fine_grained
            if self.is_multilabel is False:
                self.dev = self.dev.loc[self.dev['label'].isin(self.binary)].reset_index(drop=True)
                # map to binary then: 0, 1
                self.dev['label'] = self.dev['label'].map({1: 0, 3: 1})


        # initialize the transform if specified
        self.transform = transform

    # get len
    def __len__(self):
        if self.type == 'train':
            return len(self.train)

        if self.type == 'dev':
            return len(self.dev)

    # pull a sample of data
    def __getitem__(self, idx):
        '''
        Torch's lazy emission system
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if train, package this up
        if self.type == 'train':
            sample = {'text': self.train.text[idx],
                  'label': self.train.label[idx],
                  'idx': idx}
            if self.transform:
                sample = self.transform(sample)
            return sample

        # if dev, package this
        if self.type == 'dev':
            sample = {'text': self.dev.text[idx],
                  'label': self.dev.label[idx],
                  'idx': idx}
            if self.transform:
                sample = self.transform(sample)
            return sample


class Tokenize_Transform():
    '''
    This function tokenize transforms the data organized in SSTProcessor().

    Parameters
    ----------
    sample : dict
        A dictionary containing: (1) text, (2) labels,
        and index positions

    Returns
    -------
    sample : dict
        A dictionary containing: (1) input tokens, (2) attention masks,
        (3) token type ids, (4) labels, and (5) data set index.
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
                            max_length=64,  # set max length; SST is 64
                            truncation=True,  # truncate longer messages
                            padding='max_length',  # add padding
                            return_attention_mask=True,  # create attn. masks
                            return_tensors='pt'  # return pytorch tensors
                       )

        # package up encodings
        return {'input_ids': torch.as_tensor(encodings['input_ids'],
                                         dtype=torch.long),

                'attention_mask': torch.as_tensor(encodings['attention_mask'],
                                          dtype=torch.long),

                 'token_type_ids': torch.as_tensor(encodings['token_type_ids'],
                                                   dtype=torch.long),

                'labels': torch.as_tensor(sample['label'],
                                          dtype=torch.long),

                'idx': torch.as_tensor(sample['idx'],
                                       dtype=torch.int)}

# prepare embedding extraction
def emit_embeddings(dataloader, train_dataset, model, device, args):
    # timing metrics
    t0 = time.time()
    batch_num = 2
    embedding_batch_size = 2 # needs to perfectly divide data set
    num_documents = len(train_dataset)
    # set a manipulable var to handle any batch size
    train_len = len(train_dataset)

    with h5py.File('C:\\w266\\data2\\h5py_embeds\\sst_dev_bert_embeds.h5', 'w') as f:
        # create empty data set; [batch_sz, layers, tokens, features]
        dset = f.create_dataset('embeds', shape=(train_len, 13, args.max_seq_length, 768),
                                maxshape=(None, 13, args.max_seq_length, 768),
                                chunks=(embedding_batch_size, 13, args.max_seq_length, 768),
                                dtype=np.float32)

    with h5py.File('C:\\w266\\data2\\h5py_embeds\\sst_dev_labels.h5', 'w') as l:
        # create empty data set; [batch_sz]
        label_dset = l.create_dataset('labels', shape=(train_len,),
                                      maxshape=(None,), chunks=(embedding_batch_size,),
                                      dtype=np.int64)

    with h5py.File('C:\\w266\\data2\\h5py_embeds\\sst_dev_idx.h5', 'w') as i:
        # create empty data set; [batch_sz]
        idx_dset = i.create_dataset('idx', shape=(train_len,),
                                      maxshape=(None,), chunks=(embedding_batch_size,),
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
            # ['hidden_states'] is embeddings for all layers
            out = model(input_ids=input_ids.squeeze(1),
                        attention_mask=attn_mask.squeeze(1),
                        token_type_ids=token_type_ids.squeeze(1),
                        labels=label)

        # stack embeddings [layers, batch_sz, tokens, features]
        embeddings = torch.stack(out['hidden_states']).float()  # float32
        # swap the order to: [batch_sz, layers, tokens, features]
        # we need to do this to emit batches from h5 dataset later
        embeddings = embeddings.permute(1, 0, 2, 3).cpu().numpy()

        # add embeds to ds
        with h5py.File('C:\\w266\\data2\\h5py_embeds\\sst_dev_bert_embeds.h5', 'a') as f:
            dset = f['embeds']
            # add chunk of rows
            start = step*embedding_batch_size
            # [batch_sz, layer, tokens, features]
            dset[start:start+embedding_batch_size, :, :, :] = embeddings[:, :, :, :]
            # Create attribute with last_index value
            dset.attrs['last_index'] = (step+1)*embedding_batch_size
            # check the integrity of the embeddings
            x = f['embeds'][start:start+embedding_batch_size, :, :, :]

        # add labels to ds
        with h5py.File('C:\\w266\\data2\\h5py_embeds\\sst_dev_labels.h5', 'a') as l:
            label_dset = l['labels']
            # add chunk of rows
            start = step*embedding_batch_size
            # [batch_sz, layer, tokens, features]
            label_dset[start:start+embedding_batch_size] = label.cpu().numpy()
            # Create attribute with last_index value
            label_dset.attrs['last_index'] = (step+1)*embedding_batch_size

        # add idx to ds
        with h5py.File('C:\\w266\\data2\\h5py_embeds\\sst_dev_idx.h5', 'a') as i:
            idx_dset = i['idx']
            # add chunk of rows
            start = step*embedding_batch_size
            # [batch_sz, layer, tokens, features]
            idx_dset[start:start+embedding_batch_size] = idx.cpu().numpy()
            # Create attribute with last_index value
            idx_dset.attrs['last_index'] = (step+1)*embedding_batch_size

        batch_num += embedding_batch_size
        torch.cuda.empty_cache()

    # check data
    with h5py.File('C:\\w266\\data2\\h5py_embeds\\sst_dev_bert_embeds.h5', 'r') as f:
        print('last embed batch entry', f['embeds'].attrs['last_index'])
        print('embed shape', f['embeds'].shape)
        print('last entry:', f['embeds'][-1, :, :, :])

    with h5py.File('C:\\w266\\data2\\h5py_embeds\\sst_dev_labels.h5', 'r') as l:
        print('last embed batch entry', l['labels'].attrs['last_index'])
        print('embed shape', l['labels'].shape)
        print('last entry:', l['labels'][train_len-10: train_len])

    return None


def main():
    # training settings
    parser = ArgumentParser(description='SST2 Sentiment')
    parser.add_argument('--tokenizer', type=str,
                        default='bert-base-uncased', metavar='S',
                        help="e.g., bert-base-uncased, etc")
    parser.add_argument('--model', type=str,
                        default='bert-base-uncased', metavar='S',
                        help="e.g., bert-base-uncased, etc")
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                         help='input batch size for training (default: 2)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                         help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                         help='learning rate (default: 1e-5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                         help='random seed (default: 1)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                         help='number of CPU cores (default: 4)')
    parser.add_argument('--num-labels', type=int, default=2, metavar='N',
                         help='number of labels to classify (default: SST-2)')
    parser.add_argument('--l2', type=float, default=0.01, metavar='LR',
                         help='l2 regularization weight (default: 0.01)')
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
    dev_ds = SSTProcessor(type='dev',
                            transform=Tokenize_Transform(tokenizer=tokenizer))

    # create embed dataloader
    embed_dataloader = DataLoader(dev_ds,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                drop_last=False)


    # load the model
    model = BertForSequenceClassification.from_pretrained(args.model,
                                                          num_labels=args.num_labels,
                                                          output_hidden_states=True).to(device)

    # load weights from 1 epoch
    model.load_state_dict(torch.load('C:\\w266\\data2\\checkpoints\\BERT-sst_epoch_1.pt'))

    # create gradient scaler for mixed precision
    scaler = GradScaler()

    # export embeddings
    emit_embeddings(embed_dataloader, dev_ds, model, device, args)

# run program
if __name__ == '__main__':
    main()

#
