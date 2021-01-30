import sys
sys.path.append("C:/BERTVision/code/torch")
sys.path.append("C:\\BERTVision\\code\\torch\\bert_processors")
import torch, json, pytreebank
import pandas as pd
import numpy as np
import sys

from transformers import BertTokenizerFast

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
                                          dtype=torch.float),

                'idx': torch.as_tensor(sample['idx'],
                                       dtype=torch.int)}

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
            sample = {'text': self.train.sentence1[idx],
                      'text2': self.train.sentence2[idx],
                      'label': self.train.label[idx],
                      'idx': self.train.id[idx]}
            if self.transform:
                sample = self.transform(sample)
            return sample

        # if dev, package this
        if self.type == 'dev':
            sample = {'text': self.dev.sentence1[idx],
                      'text2': self.dev.sentence2[idx],
                      'label': self.dev.label[idx],
                      'idx': self.dev.id[idx]}
            if self.transform:
                sample = self.transform(sample)
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

class WNLI(TwoSentenceLoader):
    NAME = 'WNLI'
    def __init__(self, type, transform = None):
        '''
        Example line:
        index	sentence1	sentence2	label
        0	I stuck a pin through a carrot. When I pulled the pin out, it had a hole.	The carrot had a hole.	1
        1	John couldn't see the stage with Billy in front of him because he is so short.	John is so short.	1

        This prepares the RTE task from GLUE
        '''

        self.path = 'C:\w266\data\GLUE\Winograd NLI\WNLI'
        self.type = type
        if self.type == 'train':
            # initialize train
            self.train = pd.read_csv(self.path + '\\' + 'train.tsv', sep='\t',
                                     #names='id	qid1	qid2	question1	question2	is_duplicate'.split('\t'),
                                     encoding='latin-1',
                                     error_bad_lines=False) #SOME BAD LINES IN THIS DATA
            self.train.columns = ['id', 'sentence1', 'sentence2', 'label']

        if self.type == 'dev':
            # initialize dev
            self.dev = pd.read_csv(self.path + '\\' + 'dev.tsv', sep='\t',
                                     #names='id	qid1	qid2	question1	question2	is_duplicate'.split('\t'),
                                     encoding='latin-1')
            self.dev.columns = ['id', 'sentence1', 'sentence2', 'label']

        # initialize the transform if specified
        if transform:
            self.transform = transform
        else:
            self.transform = Tokenize_Transform()

class QNLI(TwoSentenceLoader):
    NAME = 'QNLI'
    def __init__(self, type, transform = None):
        '''
        Example line:
        index	question	sentence	label
        0	When did the third Digimon series begin?	Unlike the two seasons before it and most of the seasons that followed, Digimon Tamers takes a darker and more realistic approach to its story featuring Digimon who do not reincarnate after their deaths and more complex character development in the original Japanese.	not_entailment
        1	Which missile batteries often have individual launchers several kilometres from one another?	When MANPADS is operated by specialists, batteries may have several dozen teams deploying separately in small sections; self-propelled air defence guns may deploy in pairs.	not_entailment

        This prepares the RTE task from GLUE
        '''

        self.path = 'C:\w266\data\GLUE\Question NLI\QNLI'
        self.type = type
        if self.type == 'train':
            # initialize train
            self.train = pd.read_csv(self.path + '\\' + 'train.tsv', sep='\t',
                                     #names='id	qid1	qid2	question1	question2	is_duplicate'.split('\t'),
                                     encoding='latin-1',
                                     error_bad_lines=False) #SOME BAD LINES IN THIS DATA
            self.train.columns = ['id', 'sentence1', 'sentence2', 'label']
            self.train.label = np.where(self.train.label == 'entailment', 1, 0)

        if self.type == 'dev':
            # initialize dev
            self.dev = pd.read_csv(self.path + '\\' + 'dev.tsv', sep='\t',
                                     #names='id	qid1	qid2	question1	question2	is_duplicate'.split('\t'),
                                     encoding='latin-1')
            self.dev.columns = ['id', 'sentence1', 'sentence2', 'label']
            self.dev.columns = ['id', 'sentence1', 'sentence2', 'label']
            self.dev.label = np.where(self.dev.label == 'entailment', 1, 0)

        # initialize the transform if specified
        if transform:
            self.transform = transform
        else:
            self.transform = Tokenize_Transform()


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
            self.dev = pd.read_csv(self.path + '\\' + 'dev.tsv', sep='\t',
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
