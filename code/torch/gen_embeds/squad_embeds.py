# packages
import pathlib, time, datetime, h5py, sys
sys.path.append("C:/BERTVision/code/torch")
from argparse import ArgumentParser
from utils.collate import collate_squad_train, collate_squad_dev, collate_squad_score
from tools import AverageMeter, ProgressBar, format_time
from squad_preprocess import prepare_train_features, prepare_validation_features, postprocess_qa_predictions
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertForQuestionAnswering, AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset, load_metric


class SQuADProcessor(torch.utils.data.Dataset):
    '''
    This class uses HuggingFace's official data processing functions and
    emits them through a torch data set.

    Parameters
    ----------
    type : string
        A string used to flag conditional statements in order to use/retrieve
        the right data set.

    Returns
    -------
    sample : tensor
        A single sample of data indexed by the torch data set.
    '''

    NAME = 'SQuAD'

    def __init__(self, type):
        # flag to initialize data of choice
        self.type = type

        # if train, initialize the tokenized train data
        if self.type == 'train':
            self.train = load_dataset("squad_v2")['train'].shuffle(seed=1).map(
                prepare_train_features,
                batched=True,
                remove_columns=['answers', 'context', 'id', 'question', 'title']
                )

        # if train, initialize the tokenized dev data for training (e.g., for loss)
        if self.type == 'dev':
            self.dev = load_dataset("squad_v2")['validation'].shuffle(seed=1).map(
                prepare_train_features,
                batched=True,
                remove_columns=['answers', 'context', 'id', 'question', 'title']
                )

        # if score, initialize the tokenized dev data for validation (e.g., for metrics)
        if self.type == 'score':
            self.score = load_dataset("squad_v2")['validation'].shuffle(seed=1).map(
                prepare_validation_features,
                batched=True,
                remove_columns=['answers', 'context', 'id', 'question', 'title']
                )

        return None

    def __len__(self):
        if self.type == 'train':
            return len(self.train)
        if self.type == 'dev':
            return len(self.dev)
        if self.type == 'score':
            return len(self.score)

    def __getitem__(self, idx):
        '''
        Torch's lazy emission system
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.type == 'train':
            return (self.train[idx], idx)
        if self.type == 'dev':
            return (self.dev[idx], idx)
        if self.type == 'score':
            return (self.score[idx], idx)



# train function
def train(model, dataloader, scaler, optimizer, scheduler, device):
    pbar = ProgressBar(n_total=len(dataloader), desc='Training')
    train_loss = AverageMeter()
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        # grab items from data loader and attach to GPU
        input_ids, attn_mask, start_pos, end_pos, token_type_ids = (batch['input_ids'].to(device),
                                       batch['attention_mask'].to(device),
                                       batch['start_positions'].to(device),
                                       batch['end_positions'].to(device),
                                       batch['token_type_ids'].to(device))
        # clear gradients
        optimizer.zero_grad()
        # use mixed precision
        with autocast():
            # forward
            out = model(input_ids=input_ids,
                        attention_mask=attn_mask,
                        start_positions=start_pos,
                        end_positions=end_pos,
                        token_type_ids=token_type_ids)
        # backward
        scaler.scale(out[0]).backward()  # out[0] = loss
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        pbar(step=batch_idx, info={'loss': out[0].item()})
        train_loss.update(out[0].item(), n=1)
    train_log = {'train_loss': train_loss.avg}
    return train_log


# prepare embedding extraction
def emit_embeddings(dataloader, train_dataset, model, device, args):
    # timing metrics
    t0 = time.time()
    batch_num = args.embed_batch_size
    num_documents = len(train_dataset)

    with h5py.File('C:\\w266\\data2\\h5py_embeds\\squad_train_embeds.h5', 'w') as f:
        # create empty data set; [batch_sz, layers, tokens, features]
        dset = f.create_dataset('embeds', shape=(len(train_dataset), 13, args.max_seq_length, 768),
                                maxshape=(None, 13, args.max_seq_length, 768),
                                chunks=(args.embed_batch_size, 13, args.max_seq_length, 768),
                                dtype=np.float32)

    with h5py.File('C:\\w266\\data2\\h5py_embeds\\squad_train_start_labels.h5', 'w') as s:
        # create empty data set; [batch_sz]
        start_dset = s.create_dataset('start_ids', shape=(len(train_dataset),),
                                      maxshape=(None,), chunks=(args.embed_batch_size,),
                                      dtype=np.int64)

    with h5py.File('C:\\w266\\data2\\h5py_embeds\\squad_train_end_labels.h5', 'w') as e:
        # create empty data set; [batch_sz]
        end_dset = e.create_dataset('end_ids', shape=(len(train_dataset),),
                                      maxshape=(None,), chunks=(args.embed_batch_size,),
                                      dtype=np.int64)

    with h5py.File('C:\\w266\\data2\\h5py_embeds\\squad_train_indices.h5', 'w') as i:
        # create empty data set; [batch_sz]
        indices_dset = i.create_dataset('indices', shape=(len(train_dataset),),
                                      maxshape=(None,), chunks=(args.embed_batch_size,),
                                      dtype=np.int64)

    print('Generating embeddings for all {:,} documents...'.format(len(train_dataset)))
    for step, batch in enumerate(dataloader):
        # send necessary items to GPU
        input_ids, attn_mask, start_pos, end_pos, token_type_ids, indices = (batch['input_ids'].to(device),
                                       batch['attention_mask'].to(device),
                                       batch['start_positions'].to(device),
                                       batch['end_positions'].to(device),
                                       batch['token_type_ids'].to(device),
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
            out = model(input_ids=input_ids,
                        attention_mask=attn_mask,
                        start_positions=start_pos,
                        end_positions=end_pos,
                        token_type_ids=token_type_ids)

        # stack embeddings [layers, batch_sz, tokens, features]
        embeddings = torch.stack(out['hidden_states']).float()  # float32

        # swap the order to: [batch_sz, layers, tokens, features]
        # we need to do this to emit batches from h5 dataset later
        embeddings = embeddings.permute(1, 0, 2, 3).cpu().numpy()

        # add embeds to ds
        with h5py.File('C:\\w266\\data2\\h5py_embeds\\squad_train_embeds.h5', 'a') as f:
            dset = f['embeds']
            # add chunk of rows
            start = step*args.embed_batch_size
            # [batch_sz, layer, tokens, features]
            dset[start:start+args.embed_batch_size, :, :, :] = embeddings[:, :, :, :]
            # Create attribute with last_index value
            dset.attrs['last_index'] = (step+1)*args.embed_batch_size

        # add labels to ds
        with h5py.File('C:\\w266\\data2\\h5py_embeds\\squad_train_start_labels.h5', 'a') as s:
            start_dset = s['start_ids']
            # add chunk of rows
            start = step*args.embed_batch_size
            # [batch_sz, layer, tokens, features]
            start_dset[start:start+args.embed_batch_size] = start_pos.cpu().numpy()
            # Create attribute with last_index value
            start_dset.attrs['last_index'] = (step+1)*args.embed_batch_size

        # add labels to ds
        with h5py.File('C:\\w266\\data2\\h5py_embeds\\squad_train_end_labels.h5', 'a') as e:
            end_dset = e['end_ids']
            # add chunk of rows
            start = step*args.embed_batch_size
            # [batch_sz, layer, tokens, features]
            end_dset[start:start+args.embed_batch_size] = end_pos.cpu().numpy()
            # Create attribute with last_index value
            end_dset.attrs['last_index'] = (step+1)*args.embed_batch_size

        # add indices to ds
        with h5py.File('C:\\w266\\data2\\h5py_embeds\\squad_train_indices.h5', 'a') as i:
            indices_dset = i['indices']
            # add chunk of rows
            start = step*args.embed_batch_size
            # [batch_sz, layer, tokens, features]
            indices_dset[start:start+args.embed_batch_size] = indices.cpu().numpy()
            # Create attribute with last_index value
            indices_dset.attrs['last_index'] = (step+1)*args.embed_batch_size

        batch_num += args.embed_batch_size
        torch.cuda.empty_cache()

    # check data
    with h5py.File('C:\\w266\\data2\\h5py_embeds\\squad_train_embeds.h5', 'r') as f:
        print('last embed batch entry', f['embeds'].attrs['last_index'])
        # check the integrity of the embeddings
        x = f['embeds'][start:start+14, :, :, :]
        assert np.array_equal(x, embeddings), 'not a match'
        print('embed shape', f['embeds'].shape)
        print('last entry:', f['embeds'][-1, :, :, :])

    return None


def main():
    # training settings
    parser = ArgumentParser(description='SQuAD 2.0')
    parser.add_argument('--tokenizer', type=str,
                        default='bert-base-uncased', metavar='S',
                        help="e.g., bert-base-uncased, etc")
    parser.add_argument('--model', type=str,
                        default='bert-base-uncased', metavar='S',
                        help="e.g., bert-base-uncased, etc")
    parser.add_argument('--batch-size', type=int, default=14, metavar='N',
                         help='input batch size for training (default: 14)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                         help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=3e-5, metavar='LR',
                         help='learning rate (default: 3e-5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                         help='random seed (default: 1)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                         help='number of CPU cores (default: 4)')
    parser.add_argument('--l2', type=float, default=0.01, metavar='LR',
                         help='l2 regularization weight (default: 0.01)')
    parser.add_argument('--max-seq-length', type=int, default=384, metavar='N',
                         help='max sequence length for encoding (default: 384)')
    parser.add_argument('--warmup-proportion', type=int, default=0.1, metavar='N',
                         help='Warmup proportion (default: 0.1)')
    parser.add_argument('--embed-batch-size', type=int, default=14, metavar='N',
                         help='Embedding batch size emission (default: 14)')
    args = parser.parse_args()

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.model)

    # set squad2.0 flag
    squad_v2 = True

    # set seeds and determinism
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.amp.autocast(enabled=True)

    # tokenize / load train
    train_ds = SQuADProcessor(type='train')

    # create train dataloader
    train_dataloader = DataLoader(train_ds,
                                batch_size=args.batch_size,
                                shuffle=True,
                                collate_fn=collate_squad_train,
                                num_workers=args.num_workers,
                                drop_last=False)


    # create embed dataloader
    embed_dataloader = DataLoader(train_ds,
                                batch_size=args.embed_batch_size,
                                shuffle=True,
                                collate_fn=collate_squad_train,
                                num_workers=args.num_workers,
                                drop_last=False)

    # load the model
    model = BertForQuestionAnswering.from_pretrained(args.model).to(device)

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

    # execute the model
    best_loss = np.inf
    for epoch in range(1, args.epochs + 1):
        train_log = train(model, train_dataloader, scaler, optimizer, scheduler, device)
        if train_log['train_loss'] < best_loss:
            # torch save
            torch.save(model.state_dict(), 'C:\\BERTVision\\model_checkpoints\\BERT-QA' + '_epoch_{}.pt'.format(epoch))
            best_loss = train_log['train_loss']
        show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in train_log.items()])
        print(show_info)

    # now proceed to emit embeddings
    model = BertForQuestionAnswering.from_pretrained(args.model,
                                                     output_hidden_states=True).to(device)

    # load weights from 1 epoch
    model.load_state_dict(torch.load('C:\\BERTVision\\model_checkpoints\\BERT-QA_epoch_1.pt'))

    # export embeddings
    emit_embeddings(embed_dataloader, train_ds, model, device, args)

if __name__ == '__main__':
    main()

#
#
#
