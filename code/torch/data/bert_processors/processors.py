import torch, json, csv, sys
sys.path.append("C:/BERTVision/code/torch")
import pandas as pd
import numpy as np
import csv
from transformers import BertTokenizerFast
from datasets import load_dataset
from utils.squad_preprocess import prepare_train_features, prepare_validation_features


class Tokenize_Transform(torch.utils.data.Dataset):
    '''
    This function transforms text into tokens.

    Parameters
    ----------
    sample : dict
        A dictionary containing a sample of text.

    Returns
    -------
    sample : dict
        A dictionary containing:
        (1) input tokens,
        (2) attention masks,
        (3) token type ids,
        (4) labels, and
        (5) data set index
    '''
    def __init__(self, args, logger):
        # instantiate the tokenizer and args
        self.args = args
        self.tokenizer = BertTokenizerFast.from_pretrained(args.checkpoint)
        self.logger = logger

    def __call__(self, sample):
        # retrieve sample and unpack it
        if 'text2' not in sample:
            # tokenize as specified
            encodings = self.tokenizer(
                                       text=sample['text'],  # document to encode
                                       add_special_tokens=True,  # add '[CLS]' and '[SEP]'
                                       max_length=self.args.max_seq_length,  # set max length;
                                       truncation=True,  # truncate longer messages
                                       padding='max_length',  # add padding
                                       return_attention_mask=True,  # create attn. masks
                                       return_token_type_ids=True,  # token type ids
                           )
        # for two sentence tasks, do the following
        else:
            encodings = self.tokenizer(
                                       text=sample['text'],  # document to encode.
                                       text_pair=sample['text2'], #second sentence to encode
                                       add_special_tokens=True,  # add '[CLS]' and '[SEP]'
                                       max_length=self.args.max_seq_length,  # set max length;
                                       truncation=True,  # truncate longer messages
                                       padding='max_length',  # add padding
                                       return_attention_mask=True,  # create attn. masks
                                       return_token_type_ids=True,  # token type ids
                           )

        # package up encodings
        if self.args.model == 'STSB':
            # need to set label to float
            return {'input_ids': torch.as_tensor(encodings['input_ids'],
                                             dtype=torch.long),

                    'attention_mask': torch.as_tensor(encodings['attention_mask'],
                                              dtype=torch.long),

                     'token_type_ids': torch.as_tensor(encodings['token_type_ids'],
                                                       dtype=torch.long),

                    'labels': torch.as_tensor(sample['label'],
                                              dtype=torch.float),

                    'idx': torch.as_tensor(sample['idx'],
                                           dtype=torch.long)}
        else:
            return {'input_ids': torch.as_tensor(encodings['input_ids'],
                                             dtype=torch.long),

                    'attention_mask': torch.as_tensor(encodings['attention_mask'],
                                              dtype=torch.long),

                     'token_type_ids': torch.as_tensor(encodings['token_type_ids'],
                                                       dtype=torch.long),

                    'labels': torch.as_tensor(sample['label'],
                                              dtype=torch.long),

                    'idx': torch.as_tensor(sample['idx'],
                                           dtype=torch.long)}


class Tokenize_Transform2(torch.utils.data.Dataset):
    '''
    This function transforms text into tokens.

    Parameters
    ----------
    sample : dict
        A dictionary containing a sample of text.

    Returns
    -------
    sample : dict
        A dictionary containing:
        (1) input tokens,
        (2) attention masks,
        (3) token type ids,
        (4) labels, and
        (5) data set index
    '''
    def __init__(self, args, model = 'bert-based-uncased'):
        # instantiate the tokenizer and args
        self.args = args

        self.tokenizer = BertTokenizerFast.from_pretrained(model)
        #Don't include optional arguments are required!
        #self.logger = logger

    def __call__(self, sample):
        # retrieve sample and unpack it
        if 'text2' not in sample:
            # tokenize as specified
            encodings = self.tokenizer(
                                       text=sample['text'],  # document to encode
                                       add_special_tokens=True,  # add '[CLS]' and '[SEP]'
                                       max_length=self.args.max_seq_length,  # set max length;
                                       truncation=True,  # truncate longer messages
                                       padding='max_length',  # add padding
                                       return_attention_mask=True,  # create attn. masks
                                       return_token_type_ids=True,  # token type ids
                           )
        # for two sentence tasks, do the following
        else:
            encodings = self.tokenizer(
                                       text=sample['text'],  # document to encode.
                                       text_pair=sample['text2'], #second sentence to encode
                                       add_special_tokens=True,  # add '[CLS]' and '[SEP]'
                                       max_length=self.args.max_seq_length,  # set max length;
                                       truncation=True,  # truncate longer messages
                                       padding='max_length',  # add padding
                                       return_attention_mask=True,  # create attn. masks
                                       return_token_type_ids=True,  # token type ids
                           )

        # package up encodings
        if self.args.model == 'STSB':
            # need to set label to float
            return {'input_ids': torch.as_tensor(encodings['input_ids'],
                                             dtype=torch.long),

                    'attention_mask': torch.as_tensor(encodings['attention_mask'],
                                              dtype=torch.long),

                     'token_type_ids': torch.as_tensor(encodings['token_type_ids'],
                                                       dtype=torch.long),

                    'labels': torch.as_tensor(sample['label'],
                                              dtype=torch.float),

                    'idx': torch.as_tensor(sample['idx'],
                                           dtype=torch.long)}
        else:
            return {'input_ids': torch.as_tensor(encodings['input_ids'],
                                             dtype=torch.long),

                    'attention_mask': torch.as_tensor(encodings['attention_mask'],
                                              dtype=torch.long),

                     'token_type_ids': torch.as_tensor(encodings['token_type_ids'],
                                                       dtype=torch.long),

                    'labels': torch.as_tensor(sample['label'],
                                              dtype=torch.long),

                    'idx': torch.as_tensor(sample['idx'],
                                           dtype=torch.long)}


class OneSentenceLoader(torch.utils.data.Dataset):
    '''
    Generic class for loading datasets with 1 sentence inputs
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
            sample = {'text': self.train.sentence[idx],
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
            sample = {'text': self.dev.sentence[idx],
                      'label': self.dev.label[idx],
                      'idx': self.dev.id[idx]}
            if self.transform:
                try:
                    sample = self.transform(sample)
                except:
                    print(sample)
                    raise RuntimeError("See dev misformed sample")
            return sample


class TwoSentenceLoader(torch.utils.data.Dataset):
    '''
    Generic class for loading datasets with 2 sentence inputs
    '''
    def __init__(self):
        pass

    # get len
    def __len__(self):
        ''' get the data set length for batch purposes '''
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
    def __init__(self, type, transform=None):
        '''
        Example line:
        index	sentence1	sentence2	label
        0	No Weapons of Mass Destruction Found in Iraq Yet.	Weapons of Mass Destruction Found in Iraq.	not_entailment
        1	A place of sorrow, after Pope John Paul II died, became a place of celebration, as Roman Catholic faithful gathered in downtown Chicago to mark the installation of new Pope Benedict XVI.	Pope Benedict XVI is the new leader of the Roman Catholic Church.	entailment

        This prepares the RTE task from GLUE
        '''
        # set path for RTE
        self.path = 'C:\w266\data\GLUE\Recognizing Textual Entailment\\RTE'
        # init configurable string
        self.type = type
        # init transform if specified
        self.transform = transform

        # if type is train:
        if self.type == 'train':
            # initialize train
            self.train = pd.read_csv(self.path + '\\' + 'train.tsv', sep='\t',
                                     encoding='latin-1')
            # specify train cols
            self.train.columns = ['id', 'sentence1', 'sentence2', 'label']
            # relabel
            self.train.label = np.where(self.train.label == 'entailment', 1, 0)

        # if type is dev:
        if self.type == 'dev':
            # initialize dev
            self.dev = pd.read_csv(self.path + '\\' + 'dev.tsv', sep='\t',
                                     #names='id	qid1	qid2	question1	question2	is_duplicate'.split('\t'),
                                     encoding='latin-1')
            # specify dev cols
            self.dev.columns = ['id', 'sentence1', 'sentence2', 'label']
            # relabel
            self.dev.label = np.where(self.dev.label == 'entailment', 1, 0)




class WNLI(TwoSentenceLoader):
    NAME = 'WNLI'
    def __init__(self, type, transform=None):
        '''
        Example line:
        index	sentence1	sentence2	label
        0	I stuck a pin through a carrot. When I pulled the pin out, it had a hole.	The carrot had a hole.	1
        1	John couldn't see the stage with Billy in front of him because he is so short.	John is so short.	1

        This prepares the WNLI task from GLUE
        '''
        # set path for WNLI
        self.path = 'C:\w266\data\GLUE\Winograd NLI\WNLI'
        # init configurable string
        self.type = type
        # init transform if specified
        self.transform = transform

        # if type is train:
        if self.type == 'train':
            # initialize train
            self.train = pd.read_csv(self.path + '\\' + 'train.tsv', sep='\t',
                                     encoding='latin-1',
                                     error_bad_lines=False) #SOME BAD LINES IN THIS DATA
            # specify train cols
            self.train.columns = ['id', 'sentence1', 'sentence2', 'label']

        # if type is dev:
        if self.type == 'dev':
            # initialize dev
            self.dev = pd.read_csv(self.path + '\\' + 'dev.tsv', sep='\t',
                                     encoding='latin-1')
            # specify dev cols
            self.dev.columns = ['id', 'sentence1', 'sentence2', 'label']

        return None


class QNLI(TwoSentenceLoader):
    NAME = 'QNLI'
    def __init__(self, type, transform=None, shard=False, **args):
        '''
        Example line:
        index	question	sentence	label
        0	When did the third Digimon series begin?	Unlike the two seasons before it and most of the seasons that followed, Digimon Tamers takes a darker and more realistic approach to its story featuring Digimon who do not reincarnate after their deaths and more complex character development in the original Japanese.	not_entailment
        1	Which missile batteries often have individual launchers several kilometres from one another?	When MANPADS is operated by specialists, batteries may have several dozen teams deploying separately in small sections; self-propelled air defence guns may deploy in pairs.	not_entailment

        This prepares the QNLI task from GLUE
        '''
        # set path for QNLI
        self.path = 'C:\w266\data\GLUE\Question NLI\QNLI'
        # init configurable string
        self.type = type
        # init transform if specified
        self.transform = transform
        # init shard for sampling large ds if specified
        self.shard = shard
        # unpack useful args if given
        self.args = args
        self.seed = self.args['args'].seed
        self.shard = self.args['args'].shard

        # if type is train:
        if self.type == 'train':
            # initialize train
            self.train = pd.read_csv(self.path + '\\' + 'train.tsv', sep='\t',
                                     encoding='latin-1',
                                     error_bad_lines=False) #SOME BAD LINES IN THIS DATA
            # specify train cols
            self.train.columns = ['id', 'sentence1', 'sentence2', 'label']
            # relabel
            self.train.label = np.where(self.train.label == 'entailment', 1, 0)

            # if true, reduce train size
            if self.shard:
                self.train = self.train.sample(frac=self.shard, replace=False, random_state=self.seed).reset_index(drop=True)

        if self.type == 'dev':
            # initialize dev
            self.dev = pd.read_csv(self.path + '\\' + 'dev.tsv', sep='\t',
                                     encoding='latin-1',
                                     error_bad_lines=False)
            # specify dev cols
            self.dev.columns = ['id', 'sentence1', 'sentence2', 'label']
            # relabel
            self.dev.label = np.where(self.dev.label == 'entailment', 1, 0)

        return None



class MSR(TwoSentenceLoader):
    NAME = 'MSR'
    def __init__(self, type, transform=None):
        '''
        Example line:
        index	question	sentence	label
        0	When did the third Digimon series begin?	Unlike the two seasons before it and most of the seasons that followed, Digimon Tamers takes a darker and more realistic approach to its story featuring Digimon who do not reincarnate after their deaths and more complex character development in the original Japanese.	not_entailment
        1	Which missile batteries often have individual launchers several kilometres from one another?	When MANPADS is operated by specialists, batteries may have several dozen teams deploying separately in small sections; self-propelled air defence guns may deploy in pairs.	not_entailment

        This prepares the MSR task from GLUE
        '''
        # set path for MSR
        self.path = 'C:\w266\data\GLUE\Microsoft Research Paraphrase Corpus'
        # init configurable string
        self.type = type
        # init transform if specified
        self.transform = transform

        # if type is train:
        if self.type == 'train':
            # initialize train
            self.train = pd.read_csv(self.path + '\\' + 'msr_paraphrase_train.txt', sep='\t',
                                     encoding='latin-1',
                                     error_bad_lines=False,
                                     quoting = csv.QUOTE_NONE) #SOME BAD LINES IN THIS DATA
            # specify train cols
            self.train.columns = ['label', 'id', 'NoneField', 'sentence1', 'sentence2']

        # if type is dev:
        if self.type == 'dev':
            # initialize dev
            self.dev = pd.read_csv(self.path + '\\' + 'msr_paraphrase_test.txt', sep='\t',
                                     encoding='latin-1',
                                     error_bad_lines=False,
                                     quoting = csv.QUOTE_NONE)
            # specify dev cols
            self.dev.columns = ['label', 'id', 'NoneField', 'sentence1', 'sentence2']

        return None


class QQP(torch.utils.data.Dataset):
    NAME = 'QQP'
    def __init__(self, type, transform=None, shard=False, **args):
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
        # set path for QQPairs
        self.path = 'C:\w266\data\GLUE\Quora Question Pairs\QQP'
        # init configurable string
        self.type = type
        # init transform if specified
        self.transform = transform
        # init shard for sampling large ds if specified
        self.shard = shard
        # unpack useful args if given
        self.args = args
        self.seed = self.args['args'].seed
        self.shard = self.args['args'].shard

        # if type is train:
        if self.type == 'train':
            # initialize train
            self.train = pd.read_csv(self.path + '\\' + 'train.tsv',
                                     sep='\t',
                                     encoding='latin-1',
                                     error_bad_lines=False)

            # if true, reduce train size
            if self.shard:
                self.train = self.train.sample(frac=self.shard, replace=False, random_state=self.seed).reset_index(drop=True)

        if self.type == 'dev':
            # initialize dev
            self.dev = pd.read_csv(self.path + '\\' + 'dev.tsv',
                                   sep='\t',
                                   encoding='latin-1',
                                   error_bad_lines=False)

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
                      'label': self.dev.is_duplicate[idx],
                      'idx': self.dev.id[idx]}
            if self.transform:
                sample = self.transform(sample)
            return sample


class CoLA(torch.utils.data.Dataset):
    NAME = 'CoLA'
    def __init__(self, type, transform=None):
        '''
        https://nyu-mll.github.io/CoLA/
        Example line:
        source | label | original label | sentence
        clc95	0	*	In which way is Sandy very anxious to see if the students will be able to solve the homework problem?
        c-05	1		The book was written by John.
        c-05	0	*	Books were sent to each other by the students.
        swb04	1		She voted for herself.
        swb04	1		I saw that gas can explode.

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
        # set path for COLA
        self.path = 'C:\w266\data\GLUE\The Corpus of Linguistic Acceptability\CoLA'
        # init configurable string
        self.type = type
        # init transform if specified
        self.transform = transform

        # if type is train:
        if self.type == 'train':
            # initialize train
            self.train = pd.read_csv(self.path + '\\' + 'train.tsv',
                                     sep='\t',
                                     encoding='latin-1')
            # specify train cols
            self.train.columns = ['source', 'label', 'original_judgement', 'sentence']

        if self.type == 'dev':
            # initialize dev
            self.dev = pd.read_csv(self.path + '\\' + 'dev.tsv',
                                   sep='\t',
                                   encoding='latin-1')
            # specify dev cols
            self.dev.columns = ['source', 'label', 'original_judgement', 'sentence']

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
            sample = {'text': self.train.sentence[idx],
                      'label': self.train.label[idx],
                      'idx': idx}
            if self.transform:
                sample = self.transform(sample)
            return sample

        # if dev, package this
        if self.type == 'dev':
            sample = {'text': self.dev.sentence[idx],
                      'label': self.dev.label[idx],
                      'idx': idx}
            if self.transform:
                sample = self.transform(sample)
            return sample


class MNLI(TwoSentenceLoader):
    NAME = 'MNLI'
    def __init__(self, type, transform=None, shard=False, **args):
        '''
        Line header:
        index	promptID	pairID	genre	sentence1_binary_parse	sentence2_binary_parse	sentence1_parse	sentence2_parse	sentence1	sentence2	label1	gold_label

        This prepares the MNLI task from GLUE
        '''
        # set path for MNLI
        self.path = 'C:\w266\data\GLUE\MultiNLI (Matched and Mismatched)\MNLI'
        # init configurable string
        self.type = type
        # init transform if specified
        self.transform = transform
        # init shard for sampling large ds if specified
        self.shard = shard
        # unpack useful args if given
        self.args = args
        self.seed = self.args['args'].seed
        self.shard = self.args['args'].shard


        # if type is train:
        if self.type == 'train':
            # initialize train
            self.train = pd.read_csv(self.path + '\\' + 'train.tsv',
                                     sep='\t',
                                     encoding='latin-1',
                                     error_bad_lines=False,
                                     quoting = csv.QUOTE_NONE) #SOME BAD LINES IN THIS DATA

            # specify train cols
            self.train.columns = ['id', 'promptID', 'pairID', 'genre', 'sentence1_binary_parse', 'sentence2_binary_parse',
                                  'sentence1_parse', 'sentence2_parse', 'sentence1', 'sentence2', 'label1', 'gold_label']

            # three labels: entailment neutral contradiction
            label_map = {'neutral':0,
                         'entailment':1,
                         'contradiction':2}
            # recode
            self.train['label'] = [label_map[i] for i in self.train.gold_label]

            # change nan to empty string
            self.train['sentence2'] = self.train['sentence2'].fillna('')

            # if true, reduce train size
            if self.shard:
                self.train = self.train.sample(frac=self.shard, replace=False, random_state=self.seed).reset_index(drop=True)

        else:
            # if type is dev_matched:
            if self.type == 'dev_matched':
                # initialize dev (dev_matched set)
                self.dev = pd.read_csv(self.path + '\\' + 'dev_matched.tsv',
                                       sep='\t',
                                       encoding='latin-1',
                                       error_bad_lines=False,
                                       quoting = csv.QUOTE_NONE)

            # if type is dev_mismatched:
            if self.type == 'dev_mismatched':
                self.dev = pd.read_csv(self.path + '\\' + 'dev_mismatched.tsv',
                                       sep='\t',
                                       encoding='latin-1',
                                       error_bad_lines=False,
                                       quoting = csv.QUOTE_NONE)
            # specify dev cols
            self.dev.columns = ['id', 'promptID', 'pairID', 'genre', 'sentence1_binary_parse', 'sentence2_binary_parse',
                                  'sentence1_parse', 'sentence2_parse', 'sentence1', 'sentence2',
                                  'label1', 'label2', 'label3', 'label4', 'label5', 'gold_label']

            # three labels: entailment neutral contradiction
            label_map = {'neutral':0,
                         'entailment':1,
                         'contradiction':2}

            # recode
            self.dev['label'] = [label_map[i] for i in self.dev.gold_label]

            # change nan to empty string
            self.dev['sentence2'] = self.dev['sentence2'].fillna('')
        return None



class STSB(TwoSentenceLoader):
    NAME = 'STSB'
    def __init__(self, type, transform=None):
        '''
        Example line:
        index	genre	filename	year	old_index	source1	source2	sentence1	sentence2	score
        0	main-captions	MSRvid	2012test	0001	none	none	A plane is taking off.	An air plane is taking off.	5.000
        1	main-captions	MSRvid	2012test	0004	none	none	A man is playing a large flute.	A man is playing a flute.	3.800
        2	main-captions	MSRvid	2012test	0005	none	none	A man is spreading shreded cheese on a pizza.	A man is spreading shredded cheese on an uncooked pizza.	3.800
        3	main-captions	MSRvid	2012test	0006	none	none	Three men are playing chess.	Two men are playing chess.	2.600
        4	main-captions	MSRvid	2012test	0009	none	none	A man is playing the cello.	A man seated is playing the cello.	4.250

        This prepares the STSB task from GLUE
        '''
        # set path for STS-B
        self.path = 'C:\\w266\\data\\GLUE\\Semantic Textual Similarity Benchmark\\STS-B'
        # init configurable string
        self.type = type
        # init transform if specified
        self.transform = transform
        # specify its columns
        __cols = ['id','genre','filename','year','old_index','source1','source2','sentence1','sentence2','label']
        __dtypes = {'score':np.float16, 'index':np.int16}

        # if type is train:
        if self.type == 'train':
            # initialize train
            self.train = pd.read_csv(self.path + '\\' + 'train.tsv',
                                     sep='\t',
                                     encoding='latin-1',
                                     error_bad_lines=False,
                                     warn_bad_lines=False,
                                     dtype=__dtypes,
                                     quoting=csv.QUOTE_NONE)
            # specify train cols
            self.train.columns = __cols
            # reduce cols to just these
            self.train = self.train[['id','sentence1','sentence2','label']]
            # send label to an array
            self.train.label = self.train.label.to_numpy(dtype=np.float16)

        # if type is dev:
        if self.type == 'dev':
            # initialize dev
            self.dev = pd.read_csv(self.path + '\\' + 'dev.tsv',
                                   sep='\t',
                                   encoding='latin-1',
                                   error_bad_lines=False,
                                   warn_bad_lines=False,
                                   dtype= __dtypes,
                                   quoting=csv.QUOTE_NONE)
            # specify dev cols
            self.dev.columns = __cols
            # reduce cols to just these
            self.dev = self.dev[['id','sentence1','sentence2','label']]
            # send label to an array
            self.dev.label = self.dev.label.to_numpy(dtype=np.float16)

        return None



class SST(OneSentenceLoader):
    NAME = 'SST'
    def __init__(self, type, transform=None, shard=False, **args):
        '''
        Stanford Sentiment Treebank
        '''
        # set path for SST
        self.path = 'C:\\w266\\data\\GLUE\\The Stanford Sentiment Treebank\\SST-2'
        # init configurable string
        self.type = type
        # init transform if specified
        self.transform = transform
        # init shard for sampling large ds if specified
        self.shard = shard
        # unpack useful args if given
        self.args = args
        self.seed = self.args['args'].seed
        self.shard = self.args['args'].shard

        # SST columns
        __cols = ['label', 'sentence']
        __dtypes = {'score':np.float16, 'index':np.int16}

        # if type is train:
        if self.type == 'train':
            # initialize train
            self.train = pd.read_csv(self.path + '\\' + 'train.tsv',
                                     sep='\t',
                                     error_bad_lines=False,
                                     warn_bad_lines=False,
                                     quoting=csv.QUOTE_NONE,
                                     encoding='latin-1')
            # create id col.
            self.train['id'] = np.arange(1, len(self.train) + 1)
            # df just contains these values
            self.train = self.train[['id','sentence','label']]
            # set label
            self.train.label = self.train.label.to_numpy(dtype=np.float16)

            # if true, reduce train size
            if self.shard:
                self.train = self.train.sample(frac=self.shard, replace=False, random_state=self.seed).reset_index(drop=True)

        # if type is dev:
        if self.type == 'dev':
            # initialize dev
            self.dev = pd.read_csv(self.path + '\\' + 'dev.tsv',
                                   sep='\t',
                                   error_bad_lines=False,
                                   warn_bad_lines=False,
                                   quoting=csv.QUOTE_NONE,
                                   encoding='latin-1')
            # create id col.
            self.dev['id'] = np.arange(1, len(self.dev) + 1)
            # df just contains these values
            self.dev = self.dev[['id','sentence','label']]
            # set label
            self.dev.label = self.dev.label.to_numpy(dtype=np.float16)

        return None



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
    sample : tensor [layers, tokens, features]
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
