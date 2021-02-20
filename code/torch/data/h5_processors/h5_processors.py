# packages
import sys
sys.path.append("C:/BERTVision/code/torch")
import torch, h5py
import numpy as np

# prepare torch data set
class COLAH5Processor(torch.utils.data.Dataset):
    NAME = 'COLAH5'
    '''
    This class lazily emits batches from H5 files for deep learning.

    Parameters
    ----------
    type : string
        A string used to flag conditional statements in order to use/retrieve
        the right data set. [train or dev]

    Returns
    -------
    sample : tensor [layers, tokens, features]
        A single sample of data indexed by the torch data set.
    '''
    def __init__(self, type, args):
        self.type = type
        self.args = args

        if self.args.checkpoint == 'bert-base-uncased':
            self.file_path = 'C:\\w266\\data\\h5py_embeds\\'
        elif self.args.checkpoint == 'bert-large-uncased':
            self.file_path = 'C:\\w266\\data\\h5py_embeds\\bert_large\\'

        # todo: generalize locations
        self.train_embed_path = self.file_path + 'cola_bert_embeds.h5'
        self.train_label_path = self.file_path + 'cola_labels.h5'
        self.train_idx_path = self.file_path + 'cola_idx.h5'

        self.val_embed_path = self.file_path + 'cola_dev_bert_embeds.h5'
        self.val_label_path = self.file_path + 'cola_dev_labels.h5'
        self.val_idx_path = self.file_path + 'cola_idx.h5'

        # if train, initialize the train data
        if self.type == 'train':
            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.train_embed_path, 'r')["embeds"]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.train_label_path, 'r')["labels"]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.train_idx_path, 'r')['idx']

            with h5py.File(self.train_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

        # if train, initialize the dev data
        if self.type == 'dev':
            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.val_embed_path, 'r')["embeds"]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.val_label_path, 'r')["labels"]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.val_idx_path, 'r')['idx']

            with h5py.File(self.val_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        '''
        Torch's lazy emission system
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.type == 'train':
            # emits [layers, tokens, features]
            sample = {'embeddings': self.embeddings[idx],
                    'labels': self.labels[idx],
                    'idx': self.idx[idx]}
            return sample

        if self.type == 'dev':
            # emits [layers, tokens, features]
            sample = {'embeddings': self.embeddings[idx],
                    'labels': self.labels[idx],
                    'idx': self.idx[idx]}
            return sample

#


# prepare torch data set
class MNLIH5Processor(torch.utils.data.Dataset):
    NAME = 'MNLIH5'
    '''
    This class lazily emits batches from H5 files for deep learning.

    Parameters
    ----------
    type : string
        A string used to flag conditional statements in order to use/retrieve
        the right data set. [train or dev]

    Returns
    -------
    sample : tensor [layers, tokens, features]
        A single sample of data indexed by the torch data set.
    '''
    def __init__(self, type, args, shard=False, **kwargs):
        # init configurable string
        self.type = type
        # receive arg parser
        self.args = args
        # init shard for sampling large ds if specified
        self.shard = shard
        # set seed if given
        self.seed = kwargs

        if self.args.checkpoint == 'bert-base-uncased':
            self.file_path = 'C:\\w266\\data\\h5py_embeds\\'
        elif self.args.checkpoint == 'bert-large-uncased':
            self.file_path = 'C:\\w266\\data\\h5py_embeds\\bert_large\\'

        # todo: generalize locations
        self.train_embed_path = self.file_path + 'mnli_bert_embeds.h5'
        self.train_label_path = self.file_path + 'mnli_labels.h5'
        self.train_idx_path = self.file_path + 'mnli_idx.h5'

        self.dev_matched_embed_path = self.file_path + 'mnli_dev_matched_bert_embeds.h5'
        self.dev_matched_label_path = self.file_path + 'mnli_dev_matched_labels.h5'
        self.dev_matched_idx_path = self.file_path + 'mnli_dev_matched_idx.h5'

        self.dev_mismatched_embed_path = self.file_path + 'mnli_dev_mismatched_bert_embeds.h5'
        self.dev_mismatched_label_path = self.file_path + 'mnli_dev_mismatched_labels.h5'
        self.dev_mismatched_idx_path = self.file_path + 'mnli_dev_mismatched_idx.h5'

        # if train, initialize the train data
        if self.type == 'train' and self.shard == True:

            # initialize ds length
            with h5py.File(self.train_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

            # np seed
            np.random.seed(self.seed['seed'])
            # 15% of data set
            random_indices_len = int(0.15*self.dataset_len)
            # select randomly from indices, without replacement
            random_indices = np.random.choice(np.arange(self.dataset_len), size=random_indices_len, replace=False)
            # sort for now
            random_indices = np.sort(random_indices)

            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.train_embed_path, 'r')["embeds"][random_indices]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.train_label_path, 'r')["labels"][random_indices]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.train_idx_path, 'r')['idx'][random_indices]

        elif self.type == 'train' and self.shard == False:

            # initialize ds length
            with h5py.File(self.train_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.train_embed_path, 'r')["embeds"]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.train_label_path, 'r')["labels"]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.train_idx_path, 'r')['idx']

        # if matched, initialize the dev data
        if self.type == 'dev_matched':
            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.dev_matched_embed_path, 'r')["embeds"]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.dev_matched_label_path, 'r')["labels"]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.dev_matched_idx_path, 'r')['idx']

            with h5py.File(self.dev_matched_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

        # if matched, initialize the dev data
        if self.type == 'dev_mismatched':
            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.dev_mismatched_embed_path, 'r')["embeds"]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.dev_mismatched_label_path, 'r')["labels"]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.dev_mismatched_idx_path, 'r')['idx']

            with h5py.File(self.dev_mismatched_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        '''
        Torch's lazy emission system
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.type == 'train':
            # emits [layers, tokens, features]
            sample = {'embeddings': self.embeddings[idx],
                    'labels': self.labels[idx],
                    'idx': self.idx[idx]}
            return sample

        if self.type == 'dev_matched':
            # emits [layers, tokens, features]
            sample = {'embeddings': self.embeddings[idx],
                    'labels': self.labels[idx],
                    'idx': self.idx[idx]}
            return sample

        if self.type == 'dev_mismatched':
            # emits [layers, tokens, features]
            sample = {'embeddings': self.embeddings[idx],
                    'labels': self.labels[idx],
                    'idx': self.idx[idx]}
            return sample

#


# prepare torch data set
class MSRH5Processor(torch.utils.data.Dataset):
    NAME = 'MSRH5'
    '''
    This class lazily emits batches from H5 files for deep learning.

    Parameters
    ----------
    type : string
        A string used to flag conditional statements in order to use/retrieve
        the right data set. [train or dev]

    Returns
    -------
    sample : tensor [layers, tokens, features]
        A single sample of data indexed by the torch data set.
    '''
    def __init__(self, type, args, shard=False, **kwargs):
        # init configurable string
        self.type = type
        # receive arg parser
        self.args = args
        # init shard for sampling large ds if specified
        self.shard = shard
        # set seed if given
        self.seed = kwargs

        if self.args.checkpoint == 'bert-base-uncased':
            self.file_path = 'C:\\w266\\data\\h5py_embeds\\'
        elif self.args.checkpoint == 'bert-large-uncased':
            self.file_path = 'C:\\w266\\data\\h5py_embeds\\bert_large\\'

        # todo: generalize locations
        self.train_embed_path = self.file_path + 'msr_bert_embeds.h5'
        self.train_label_path = self.file_path + 'msr_labels.h5'
        self.train_idx_path = self.file_path + 'msr_idx.h5'

        self.val_embed_path = self.file_path + 'msr_dev_bert_embeds.h5'
        self.val_label_path = self.file_path + 'msr_dev_labels.h5'
        self.val_idx_path = self.file_path + 'msr_idx.h5'

        # if train, initialize the train data
        if self.type == 'train' and self.shard == True:

            # initialize ds length
            with h5py.File(self.train_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

            # np seed
            np.random.seed(self.seed['seed'])
            # 15% of data set
            random_indices_len = int(0.15*self.dataset_len)
            # select randomly from indices, without replacement
            random_indices = np.random.choice(np.arange(self.dataset_len), size=random_indices_len, replace=False)
            # sort for now
            random_indices = np.sort(random_indices)

            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.train_embed_path, 'r')["embeds"][random_indices]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.train_label_path, 'r')["labels"][random_indices]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.train_idx_path, 'r')['idx'][random_indices]

            # run it again with sharded ds length
            with h5py.File(self.train_embed_path, 'r') as file:
                self.dataset_len = len(self.embeddings)

        elif self.type == 'train' and self.shard == False:

            # initialize ds length
            with h5py.File(self.train_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.train_embed_path, 'r')["embeds"]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.train_label_path, 'r')["labels"]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.train_idx_path, 'r')['idx']


        # if train, initialize the dev data
        if self.type == 'dev':
            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.val_embed_path, 'r')["embeds"]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.val_label_path, 'r')["labels"]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.val_idx_path, 'r')['idx']

            with h5py.File(self.val_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        '''
        Torch's lazy emission system
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.type == 'train':
            # emits [layers, tokens, features]
            sample = {'embeddings': self.embeddings[idx],
                    'labels': self.labels[idx],
                    'idx': self.idx[idx]}
            return sample

        if self.type == 'dev':
            # emits [layers, tokens, features]
            sample = {'embeddings': self.embeddings[idx],
                    'labels': self.labels[idx],
                    'idx': self.idx[idx]}
            return sample


#


# prepare torch data set
class QNLIH5Processor(torch.utils.data.Dataset):
    NAME = 'QNLIH5'
    '''
    This class lazily emits batches from H5 files for deep learning.

    Parameters
    ----------
    type : string
        A string used to flag conditional statements in order to use/retrieve
        the right data set. [train or dev]

    Returns
    -------
    sample : tensor [layers, tokens, features]
        A single sample of data indexed by the torch data set.
    '''
    def __init__(self, type, args, shard=False, **kwargs):
        # init configurable string
        self.type = type
        # receive arg parser
        self.args = args
        # init shard for sampling large ds if specified
        self.shard = shard
        # set seed if given
        self.seed = kwargs


        if self.args.checkpoint == 'bert-base-uncased':
            self.file_path = 'C:\\w266\\data\\h5py_embeds\\'
        elif self.args.checkpoint == 'bert-large-uncased':
            self.file_path = 'C:\\w266\\data\\h5py_embeds\\bert_large\\'

        # h5 locs
        self.train_embed_path = self.file_path + 'qnli_bert_embeds.h5'
        self.train_label_path = self.file_path + 'qnli_labels.h5'
        self.train_idx_path = self.file_path + 'qnli_idx.h5'

        self.val_embed_path = self.file_path + 'qnli_dev_bert_embeds.h5'
        self.val_label_path = self.file_path + 'qnli_dev_labels.h5'
        self.val_idx_path = self.file_path + 'qnli_idx.h5'

        # if train, initialize the train data
        if self.type == 'train' and self.shard == True:
            # initialize ds length
            with h5py.File(self.train_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

            # np seed
            np.random.seed(self.seed['seed'])
            # 15% of data set
            random_indices_len = int(0.15*self.dataset_len)
            # select randomly from indices, without replacement
            random_indices = np.random.choice(np.arange(self.dataset_len), size=random_indices_len, replace=False)
            # sort for now
            random_indices = np.sort(random_indices)

            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.train_embed_path, 'r')["embeds"][random_indices]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.train_label_path, 'r')["labels"][random_indices]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.train_idx_path, 'r')['idx'][random_indices]

        elif self.type == 'train' and self.shard == False:
            # initialize ds length
            with h5py.File(self.train_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.train_embed_path, 'r')["embeds"]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.train_label_path, 'r')["labels"]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.train_idx_path, 'r')['idx']

        # if train, initialize the dev data
        if self.type == 'dev':
            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.val_embed_path, 'r')["embeds"]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.val_label_path, 'r')["labels"]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.val_idx_path, 'r')['idx']

            with h5py.File(self.val_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        '''
        Torch's lazy emission system
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.type == 'train':
            # emits [layers, tokens, features]
            sample = {'embeddings': self.embeddings[idx],
                    'labels': self.labels[idx],
                    'idx': self.idx[idx]}
            return sample

        if self.type == 'dev':
            # emits [layers, tokens, features]
            sample = {'embeddings': self.embeddings[idx],
                    'labels': self.labels[idx],
                    'idx': self.idx[idx]}
            return sample


#


# prepare torch data set
class QQPH5Processor(torch.utils.data.Dataset):
    NAME = 'QQPH5'
    '''
    This class lazily emits batches from H5 files for deep learning.

    Parameters
    ----------
    type : string
        A string used to flag conditional statements in order to use/retrieve
        the right data set. [train or dev]

    Returns
    -------
    sample : tensor [layers, tokens, features]
        A single sample of data indexed by the torch data set.
    '''
    def __init__(self, type, args, shard=False, **kwargs):
        # init configurable string
        self.type = type
        # receive arg parser
        self.args = args
        # init shard for sampling large ds if specified
        self.shard = shard
        # set seed if given
        self.seed = kwargs

        if self.args.checkpoint == 'bert-base-uncased':
            self.file_path = 'C:\\w266\\data\\h5py_embeds\\'
        elif self.args.checkpoint == 'bert-large-uncased':
            self.file_path = 'C:\\w266\\data\\h5py_embeds\\bert_large\\'

        # h5 locs
        self.train_embed_path = self.file_path + 'qqpairs_bert_embeds.h5'
        self.train_label_path = self.file_path + 'qqpairs_labels.h5'
        self.train_idx_path = self.file_path + 'qqpairs_idx.h5'

        self.val_embed_path = self.file_path + 'qqpairs_dev_bert_embeds.h5'
        self.val_label_path = self.file_path + 'qqpairs_dev_labels.h5'
        self.val_idx_path = self.file_path + 'qqpairs_idx.h5'

        # if train, initialize the train data
        if self.type == 'train' and self.shard == True:

            # initialize ds length
            with h5py.File(self.train_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

            # np seed
            np.random.seed(self.seed['seed'])
            # 15% of data set
            random_indices_len = int(0.15*self.dataset_len)
            # select randomly from indices, without replacement
            random_indices = np.random.choice(np.arange(self.dataset_len), size=random_indices_len, replace=False)
            # sort for now
            random_indices = np.sort(random_indices)

            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.train_embed_path, 'r')["embeds"][random_indices]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.train_label_path, 'r')["labels"][random_indices]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.train_idx_path, 'r')['idx'][random_indices]

        elif self.type == 'train' and self.shard == False:

            # initialize ds length
            with h5py.File(self.train_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.train_embed_path, 'r')["embeds"]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.train_label_path, 'r')["labels"]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.train_idx_path, 'r')['idx']

        # if train, initialize the dev data
        if self.type == 'dev':
            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.val_embed_path, 'r')["embeds"]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.val_label_path, 'r')["labels"]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.val_idx_path, 'r')['idx']

            with h5py.File(self.val_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        '''
        Torch's lazy emission system
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.type == 'train':
            # emits [layers, tokens, features]
            sample = {'embeddings': self.embeddings[idx],
                    'labels': self.labels[idx],
                    'idx': self.idx[idx]}
            return sample

        if self.type == 'dev':
            # emits [layers, tokens, features]
            sample = {'embeddings': self.embeddings[idx],
                    'labels': self.labels[idx],
                    'idx': self.idx[idx]}
            return sample


#


# prepare torch data set
class RTEH5Processor(torch.utils.data.Dataset):
    NAME = 'RTEH5'
    '''
    This class lazily emits batches from H5 files for deep learning.

    Parameters
    ----------
    type : string
        A string used to flag conditional statements in order to use/retrieve
        the right data set. [train or dev]

    Returns
    -------
    sample : tensor [layers, tokens, features]
        A single sample of data indexed by the torch data set.
    '''
    def __init__(self, type, args):
        self.type = type
        self.args = args

        if self.args.checkpoint == 'bert-base-uncased':
            self.file_path = 'C:\\w266\\data\\h5py_embeds\\'
        elif self.args.checkpoint == 'bert-large-uncased':
            self.file_path = 'C:\\w266\\data\\h5py_embeds\\bert_large\\'

        # todo: generalize locations
        self.train_embed_path = self.file_path + 'rte_bert_embeds.h5'
        self.train_label_path = self.file_path + 'rte_labels.h5'
        self.train_idx_path = self.file_path + 'rte_idx.h5'

        self.val_embed_path = self.file_path + 'rte_dev_bert_embeds.h5'
        self.val_label_path = self.file_path + 'rte_dev_labels.h5'
        self.val_idx_path = self.file_path + 'rte_idx.h5'

        # if train, initialize the train data
        if self.type == 'train':
            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.train_embed_path, 'r')["embeds"]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.train_label_path, 'r')["labels"]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.train_idx_path, 'r')['idx']

            with h5py.File(self.train_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

        # if train, initialize the dev data
        if self.type == 'dev':
            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.val_embed_path, 'r')["embeds"]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.val_label_path, 'r')["labels"]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.val_idx_path, 'r')['idx']

            with h5py.File(self.val_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        '''
        Torch's lazy emission system
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.type == 'train':
            # emits [layers, tokens, features]
            sample = {'embeddings': self.embeddings[idx],
                    'labels': self.labels[idx],
                    'idx': self.idx[idx]}
            return sample

        if self.type == 'dev':
            # emits [layers, tokens, features]
            sample = {'embeddings': self.embeddings[idx],
                    'labels': self.labels[idx],
                    'idx': self.idx[idx]}
            return sample


# prepare torch data set
class SQuADH5Processor(torch.utils.data.Dataset):
    '''
    This class lazily emits batches from H5 files for deep learning.

    Parameters
    ----------
    type : string
        A string used to flag conditional statements in order to use/retrieve
        the right data set. [train or dev]

    Returns
    -------
    sample : tensor [layers, tokens, features]
        A single sample of data indexed by the torch data set.
    '''

    NAME = 'SQuADH5'

    def __init__(self, type):
        self.type = type

        # todo: generalize locations
        self.train_embed_path = 'C:\\w266\data\\h5py_embeds\\squad_train_embeds.h5'
        self.train_start_path = 'C:\\w266\data\\h5py_embeds\\squad_train_start_labels.h5'
        self.train_end_path = 'C:\\w266\data\\h5py_embeds\\squad_train_end_labels.h5'
        self.train_idx_path = 'C:\\w266\data\\h5py_embeds\\squad_train_indices.h5'

        self.val_embed_path = 'C:\\w266\data\\h5py_embeds\\squad_dev_embeds.h5'
        self.val_start_path = 'C:\\w266\data\\h5py_embeds\\squad_dev_start_labels.h5'
        self.val_end_path = 'C:\\w266\data\\h5py_embeds\\squad_dev_end_labels.h5'
        self.val_idx_path = 'C:\\w266\data\\h5py_embeds\\squad_dev_indices.h5'

        # if train, initialize the train data
        if self.type == 'train':
            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.train_embed_path, 'r')["embeds"]
            # start ids are shaped: [batch_sz, ]
            self.start = h5py.File(self.train_start_path, 'r')["start_ids"]
            # end ids are shaped: [batch_sz, ]
            self.end = h5py.File(self.train_end_path, 'r')["end_ids"]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.train_idx_path, 'r')['indices']

            with h5py.File(self.train_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

        # if train, initialize the dev data
        if self.type == 'dev':
            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.val_embed_path, 'r')["embeds"]
            # start ids are shaped: [batch_sz, ]
            self.start = h5py.File(self.val_start_path, 'r')["start_ids"]
            # end ids are shaped: [batch_sz, ]
            self.end = h5py.File(self.val_end_path, 'r')["end_ids"]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.val_idx_path, 'r')['indices']

            with h5py.File(self.val_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        '''
        Torch's lazy emission system
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.type == 'train':
            # emits [layers, tokens, features]
            sample = {'embeddings': self.embeddings[idx],
                    'start_ids': self.start[idx],
                    'end_ids': self.end[idx],
                    'idx': self.idx[idx]}
            return sample

        if self.type == 'dev':
            # emits [layers, tokens, features]
            sample = {'embeddings': self.embeddings[idx],
                    'start_ids': self.start[idx],
                    'end_ids': self.end[idx],
                    'idx': self.idx[idx]}
            return sample

#

# prepare torch data set
class SSTH5Processor(torch.utils.data.Dataset):
    NAME = 'SSTH5'
    '''
    This class lazily emits batches from H5 files for deep learning.

    Parameters
    ----------
    type : string
        A string used to flag conditional statements in order to use/retrieve
        the right data set. [train or dev]

    Returns
    -------
    sample : tensor [layers, tokens, features]
        A single sample of data indexed by the torch data set.
    '''
    def __init__(self, type, args):
        self.type = type
        self.args = args

        if self.args.checkpoint == 'bert-base-uncased':
            self.file_path = 'C:\\w266\\data\\h5py_embeds\\'
        elif self.args.checkpoint == 'bert-large-uncased':
            self.file_path = 'C:\\w266\\data\\h5py_embeds\\bert_large\\'

        # todo: generalize locations
        self.train_embed_path = self.file_path + 'sst_bert_embeds.h5'
        self.train_label_path = self.file_path + 'sst_labels.h5'
        self.train_idx_path = self.file_path + 'sst_idx.h5'

        self.val_embed_path = self.file_path + 'sst_dev_bert_embeds.h5'
        self.val_label_path = self.file_path + 'sst_dev_labels.h5'
        self.val_idx_path = self.file_path + 'sst_idx.h5'

        # if train, initialize the train data
        if self.type == 'train':
            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.train_embed_path, 'r')["embeds"]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.train_label_path, 'r')["labels"]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.train_idx_path, 'r')['idx']

            with h5py.File(self.train_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

        # if train, initialize the dev data
        if self.type == 'dev':
            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.val_embed_path, 'r')["embeds"]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.val_label_path, 'r')["labels"]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.val_idx_path, 'r')['idx']

            with h5py.File(self.val_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        '''
        Torch's lazy emission system
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.type == 'train':
            # emits [layers, tokens, features]
            sample = {'embeddings': self.embeddings[idx],
                    'labels': self.labels[idx],
                    'idx': self.idx[idx]}
            return sample

        if self.type == 'dev':
            # emits [layers, tokens, features]
            sample = {'embeddings': self.embeddings[idx],
                    'labels': self.labels[idx],
                    'idx': self.idx[idx]}
            return sample


# prepare torch data set
class STSBH5Processor(torch.utils.data.Dataset):
    NAME = 'STSBH5'
    '''
    This class lazily emits batches from H5 files for deep learning.

    Parameters
    ----------
    type : string
        A string used to flag conditional statements in order to use/retrieve
        the right data set. [train or dev]

    Returns
    -------
    sample : tensor [layers, tokens, features]
        A single sample of data indexed by the torch data set.
    '''
    def __init__(self, type, args):
        self.type = type
        self.args = args

        if self.args.checkpoint == 'bert-base-uncased':
            self.file_path = 'C:\\w266\\data\\h5py_embeds\\'
        elif self.args.checkpoint == 'bert-large-uncased':
            self.file_path = 'C:\\w266\\data\\h5py_embeds\\bert_large\\'

        # todo: generalize locations
        self.train_embed_path = self.file_path + 'stsb_bert_embeds.h5'
        self.train_label_path = self.file_path + 'stsb_labels.h5'
        self.train_idx_path = self.file_path + 'stsb_idx.h5'

        self.val_embed_path = self.file_path + 'stsb_dev_bert_embeds.h5'
        self.val_label_path = self.file_path + 'stsb_dev_labels.h5'
        self.val_idx_path = self.file_path + 'stsb_idx.h5'

        # if train, initialize the train data
        if self.type == 'train':
            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.train_embed_path, 'r')["embeds"]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.train_label_path, 'r')["labels"]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.train_idx_path, 'r')['idx']

            with h5py.File(self.train_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

        # if train, initialize the dev data
        if self.type == 'dev':
            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.val_embed_path, 'r')["embeds"]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.val_label_path, 'r')["labels"]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.val_idx_path, 'r')['idx']

            with h5py.File(self.val_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        '''
        Torch's lazy emission system
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.type == 'train':
            # emits [layers, tokens, features]
            sample = {'embeddings': self.embeddings[idx],
                    'labels': self.labels[idx],
                    'idx': self.idx[idx]}
            return sample

        if self.type == 'dev':
            # emits [layers, tokens, features]
            sample = {'embeddings': self.embeddings[idx],
                    'labels': self.labels[idx],
                    'idx': self.idx[idx]}
            return sample


#
# prepare torch data set
class WNLIH5Processor(torch.utils.data.Dataset):
    NAME = 'WNLIH5'
    '''
    This class lazily emits batches from H5 files for deep learning.

    Parameters
    ----------
    type : string
        A string used to flag conditional statements in order to use/retrieve
        the right data set. [train or dev]

    Returns
    -------
    sample : tensor [layers, tokens, features]
        A single sample of data indexed by the torch data set.
    '''
    def __init__(self, type, args):
        self.type = type
        self.args = args

        if self.args.checkpoint == 'bert-base-uncased':
            self.file_path = 'C:\\w266\\data\\h5py_embeds\\'
        elif self.args.checkpoint == 'bert-large-uncased':
            self.file_path = 'C:\\w266\\data\\h5py_embeds\\bert_large\\'

        # todo: generalize locations
        self.train_embed_path = self.file_path + 'wnli_bert_embeds.h5'
        self.train_label_path = self.file_path + 'wnli_labels.h5'
        self.train_idx_path = self.file_path + 'wnli_idx.h5'

        self.val_embed_path = self.file_path + 'wnli_dev_bert_embeds.h5'
        self.val_label_path = self.file_path + 'wnli_dev_labels.h5'
        self.val_idx_path = self.file_path + 'wnli_idx.h5'

        # if train, initialize the train data
        if self.type == 'train':
            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.train_embed_path, 'r')["embeds"]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.train_label_path, 'r')["labels"]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.train_idx_path, 'r')['idx']

            with h5py.File(self.train_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

        # if train, initialize the dev data
        if self.type == 'dev':
            # embeds are shaped: [batch_sz, layers, tokens, features]
            self.embeddings = h5py.File(self.val_embed_path, 'r')["embeds"]
            # labels are shaped: [batch_sz, ]
            self.labels = h5py.File(self.val_label_path, 'r')["labels"]
            # idx ids are shaped: [batch_sz, ]
            self.idx = h5py.File(self.val_idx_path, 'r')['idx']

            with h5py.File(self.val_embed_path, 'r') as file:
                self.dataset_len = len(file["embeds"])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        '''
        Torch's lazy emission system
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.type == 'train':
            # emits [layers, tokens, features]
            sample = {'embeddings': self.embeddings[idx],
                    'labels': self.labels[idx],
                    'idx': self.idx[idx]}
            return sample

        if self.type == 'dev':
            # emits [layers, tokens, features]
            sample = {'embeddings': self.embeddings[idx],
                    'labels': self.labels[idx],
                    'idx': self.idx[idx]}
            return sample

#
