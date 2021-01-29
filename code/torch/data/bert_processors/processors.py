import sys
sys.path.append("C:/BERTVision/code/torch")
import torch, json, pytreebank
import pandas as pd
from utils import Tokenize_Transform

class QQPairs(torch.utils.data.Dataset):
    '''
    Example line:
    id	qid1	qid2	question1	question2	is_duplicate
    133273	213221	213222	How is the life of a math student? Could you describe your own experiences?	Which level of prepration is enough for the exam jlpt5?	0
    402555	536040	536041	How do I control my horny emotions?	How do you control your horniness?	1

    This prepares the Quora Question Pairs GLUE Task

    Parameters
    ----------
    transform : optionable, callable flag
        Whether or not we need to tokenize transform the data

    Returns
    -------
    sample : dict
        A dictionary containing: (1) text, (2) text2, (3) labels,
        and index positions
    '''

    NAME = 'QQPairs'

    def __init__(self, type, transform=None):
        # set path for data
        self.path = 'C:\w266\data\GLUE\Quora Question Pairs\QQP'
        self.type = type
        if self.type == 'train':
            # initialize train
            self.train = pd.read_csv(self.path + '\\' + 'train.tsv', sep='\t',
                                     #names='id	qid1	qid2	question1	question2	is_duplicate'.split('\t'),
                                     encoding='latin-1')
        if self.type == 'dev':
            # initialize dev
            self.dev = pd.read_csv(self.path + '\\' + 'dev.txt', sep='\t',
                                     #names='id	qid1	qid2	question1	question2	is_duplicate'.split('\t'),
                                     encoding='latin-1')

        # initialize the transform if specified
        if transform:
            self.transform = transform
        else:
            self.transform = Tokenize_Transform()

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
            sample = {'text': self.train.question1[idx],
                      'text2': self.train.question2[idx],
                      'label': self.train.is_duplicate[idx],
                      'idx': self.train.id[idx]}
            if self.transform:
                sample = self.transform(sample)
            return sample

        # if dev, package this
        if self.type == 'dev':
            sample = {'text': self.dev.question1[idx],
                      'text2': self.dev.question2[idx],
                      'is_duplicate': self.dev.is_duplicate[idx],
                      'idx': self.dev.id[idx]}
            if self.transform:
                sample = self.transform(sample)
            return sample
