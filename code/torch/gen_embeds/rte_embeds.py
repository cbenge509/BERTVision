# packages
from argparse import ArgumentParser
import sys
sys.path.append("C:/BERTVision/code/torch")
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

# prepare torch data set
class TwoSentenceLoader(torch.utils.data.Dataset):
    '''
    Generic class for loading datasets with 2 sentence inputs
    '''

    def __init__(self):
        # set path for data
        pass

    # get len
    def __len__(self):
        if self.type == 'train':
            return len(self.train)

        if 'dev' in self.type:
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
            sample = {'text': self.train.sentence1[idx],
                      'text2': self.train.sentence2[idx],
                      'label': self.train.label[idx],
                      'idx': self.train.id[idx]}
            if self.transform:
                try:
                    sample = self.transform(sample)
                except:
                    print(sample)
                    raise RuntimeError("See train misformed sample")
            return sample

        # if dev, package this
        if 'dev' in self.type:
            sample = {'text': self.dev.sentence1[idx],
                      'text2': self.dev.sentence2[idx],
                      'label': self.dev.label[idx],
                      'idx': self.dev.id[idx]}
            if self.transform:
                try:
                    sample = self.transform(sample)
                except:
                    print(sample)
                    raise RuntimeError("See dev misformed sample")
            return sample



class RTE(TwoSentenceLoader):
    NAME = 'RTE'
    def __init__(self, type, transform = None):
        '''
        Example line:
        index	sentence1	sentence2	label
        0	No Weapons of Mass Destruction Found in Iraq Yet.	Weapons of Mass Destruction Found in Iraq.	not_entailment
        1	A place of sorrow, after Pope John Paul II died, became a place of celebration, as Roman Catholic faithful gathered in downtown Chicago to mark the installation of new Pope Benedict XVI.	Pope Benedict XVI is the new leader of the Roman Catholic Church.	entailment

        This prepares the RTE task from GLUE
        '''

        self.path = 'C:\w266\data\GLUE\Recognizing Textual Entailment\\RTE'
        self.type = type
        if self.type == 'train':
            # initialize train
            self.train = pd.read_csv(self.path + '\\' + 'train.tsv', sep='\t',
                                     #names='id	qid1	qid2	question1	question2	is_duplicate'.split('\t'),
                                     encoding='latin-1')
            self.train.columns = ['id', 'sentence1', 'sentence2', 'label']
            self.train.label = np.where(self.train.label == 'entailment', 1, 0)

        if self.type == 'dev':
            # initialize dev
            self.dev = pd.read_csv(self.path + '\\' + 'dev.tsv', sep='\t',
                                     #names='id	qid1	qid2	question1	question2	is_duplicate'.split('\t'),
                                     encoding='latin-1')
            self.dev.columns = ['id', 'sentence1', 'sentence2', 'label']
            self.dev.label = np.where(self.dev.label == 'entailment', 1, 0)

        # initialize the transform if specified
        if transform:
            self.transform = transform
        else:
            self.transform = Tokenize_Transform()



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
    def __init__(self):
        # instantiate the tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # retrieve sample and unpack it
    def __call__(self, sample):
        # transform text to input ids and attn masks

        if 'text2' not in sample:
            encodings = self.tokenizer(
                                sample['text'],  # document to encode.
                                add_special_tokens=True,  # add '[CLS]' and '[SEP]'
                                max_length=64,  # set max length; SST is 64
                                truncation=True,  # truncate longer messages
                                padding='max_length',  # add padding
                                return_attention_mask=True,  # create attn. masks
                                return_tensors='pt'  # return pytorch tensors
                           )
        else:
            encodings = self.tokenizer(
                                sample['text'],  # document to encode.
                                sample['text2'], #second sentence to encode
                                add_special_tokens=True,  # add '[CLS]' and '[SEP]'
                                max_length=512,  # set max length; SST is 64
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


# train function
def train(model, dataloader, scaler, optimizer, scheduler, device):
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

        pred = out['logits'].argmax(dim=1, keepdim=True)
        correct = pred.eq(label.view_as(pred)).sum().item()
        f1 = f1_score(pred.cpu().numpy(), label.cpu().numpy(), average='weighted')
        scaler.scale(out['loss']).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
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
    batch_num = args.embed_batch_size  # needs to perfectly divide data set
    num_documents = len(train_dataset)

    with h5py.File('C:\\w266\\data2\\h5py_embeds\\rte_bert_embeds.h5', 'w') as f:
        # create empty data set; [batch_sz, layers, tokens, features]
        dset = f.create_dataset('embeds', shape=(len(train_dataset), 13, args.max_seq_length, 768),
                                maxshape=(None, 13, args.max_seq_length, 768),
                                chunks=(args.embed_batch_size, 13, args.max_seq_length, 768),
                                dtype=np.float32)

    with h5py.File('C:\\w266\\data2\\h5py_embeds\\rte_labels.h5', 'w') as l:
        # create empty data set; [batch_sz]
        label_dset = l.create_dataset('labels', shape=(len(train_dataset),),
                                      maxshape=(None,), chunks=(args.embed_batch_size,),
                                      dtype=np.int64)

    with h5py.File('C:\\w266\\data2\\h5py_embeds\\rte_idx.h5', 'w') as i:
        # create empty data set; [batch_sz]
        idx_dset = i.create_dataset('idx', shape=(len(train_dataset),),
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
        with h5py.File('C:\\w266\\data2\\h5py_embeds\\rte_bert_embeds.h5', 'a') as f:
            dset = f['embeds']
            # add chunk of rows
            start = step*args.embed_batch_size
            # [batch_sz, layer, tokens, features]
            dset[start:start+args.embed_batch_size, :, :, :] = embeddings[:, :, :, :]
            # Create attribute with last_index value
            dset.attrs['last_index'] = (step+1)*args.embed_batch_size
            # check the integrity of the embeddings
            x = f['embeds'][start:start+args.embed_batch_size, :, :, :]

        # add labels to ds
        with h5py.File('C:\\w266\\data2\\h5py_embeds\\rte_labels.h5', 'a') as l:
            label_dset = l['labels']
            # add chunk of rows
            start = step*args.embed_batch_size
            # [batch_sz, layer, tokens, features]
            label_dset[start:start+args.embed_batch_size] = label.cpu().numpy()
            # Create attribute with last_index value
            label_dset.attrs['last_index'] = (step+1)*args.embed_batch_size

        # add idx to ds
        with h5py.File('C:\\w266\\data2\\h5py_embeds\\rte_idx.h5', 'a') as i:
            idx_dset = i['idx']
            # add chunk of rows
            start = step*args.embed_batch_size
            # [batch_sz, layer, tokens, features]
            idx_dset[start:start+args.embed_batch_size] = idx.cpu().numpy()
            # Create attribute with last_index value
            idx_dset.attrs['last_index'] = (step+1)*args.embed_batch_size

        batch_num += args.embed_batch_size
        torch.cuda.empty_cache()

    # check data
    with h5py.File('C:\\w266\\data2\\h5py_embeds\\rte_bert_embeds.h5', 'r') as f:
        print('last embed batch entry', f['embeds'].attrs['last_index'])
        print('embed shape', f['embeds'].shape)
        print('last entry:', f['embeds'][-1, :, :, :])

    with h5py.File('C:\\w266\\data2\\h5py_embeds\\rte_labels.h5', 'r') as l:
        print('last embed batch entry', l['labels'].attrs['last_index'])
        print('embed shape', l['labels'].shape)
        print('last entry:', l['labels'][len(train_dataset)-10: len(train_dataset)])

    return None



def main():
    # training settings
    parser = ArgumentParser(description='RTE - Recognizing Textual Entailment')
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
                         help='learning rate (default: 3e-5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                         help='random seed (default: 1)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                         help='number of CPU cores (default: 4)')
    parser.add_argument('--num-labels', type=int, default=2, metavar='N',
                         help='number of labels to classify (default: 2)')
    parser.add_argument('--l2', type=float, default=0.01, metavar='LR',
                         help='l2 regularization weight (default: 0.01)')
    parser.add_argument('--max-seq-length', type=int, default=512, metavar='N',
                         help='max sequence length for encoding (default: 512)')
    parser.add_argument('--warmup-proportion', type=int, default=0.1, metavar='N',
                         help='Warmup proportion (default: 0.1)')
    parser.add_argument('--embed-batch-size', type=int, default=1, metavar='N',
                         help='Embedding batch size emission; (default: 1)')
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
    train_ds = RTE(type='train')

    # create training dataloader
    train_dataloader = DataLoader(train_ds,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  drop_last=False)

    # create embed dataloader
    embed_dataloader = DataLoader(train_ds,
                                batch_size=args.embed_batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                drop_last=False)


    # load the model
    model = BertForSequenceClassification.from_pretrained(args.model,
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

    # train
    best_loss = np.inf
    for epoch in range(1, epochs + 1):
        train_log = train(model, train_dataloader, scaler, optimizer, scheduler, device)
        logs = dict(train_log)
        if logs['loss'] < best_loss:
            # torch save
            torch.save(model.state_dict(), 'C:\\w266\\data2\\checkpoints\\BERT-RTE' + '_epoch_{}.pt'.format(epoch))
            best_loss = logs['loss']
        show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        print(show_info)

    # now proceed to emit embeddings
    model = BertForSequenceClassification.from_pretrained(args.model,
                                                          num_labels=args.num_labels,
                                                          output_hidden_states=True).to(device)
    # load weights from 1 epoch
    model.load_state_dict(torch.load('C:\\w266\\data2\\checkpoints\\BERT-RTE_epoch_1.pt'))

    # export embeddings
    emit_embeddings(embed_dataloader, train_ds, model, device, args)

# run program
if __name__ == '__main__':
    main()

#