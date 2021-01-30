# packages
import sys
sys.path.append("C:/BERTVision/code/torch")
import torch
from datasets import load_dataset
from utils.squad_preprocess import prepare_train_features, prepare_validation_features


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
#
