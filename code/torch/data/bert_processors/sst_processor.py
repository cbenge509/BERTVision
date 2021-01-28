# packages
import sys
sys.path.append("C:/BERTVision/code/torch")
import torch, json, pytreebank
import pandas as pd
from transformers import BertTokenizerFast


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

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

    def __init__(self, args, type, transform=None):
        # set path for data
        self.path = 'C:\\BERTVision\\code\\torch\\data\\data_sets\\sst\\trees'
        # initialize flag; set false for binary classification
        self.is_multilabel = args.is_multilabel
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



#
