# packages
import sys
sys.path.append("C:/BERTVision")
import torch, h5py


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
        self.train_embed_path = 'C:\\w266\data2\\h5py_embeds\\squad_train_embeds.h5'
        self.train_start_path = 'C:\\w266\data2\\h5py_embeds\\squad_train_start_labels.h5'
        self.train_end_path = 'C:\\w266\data2\\h5py_embeds\\squad_train_end_labels.h5'
        self.train_idx_path = 'C:\\w266\data2\\h5py_embeds\\squad_train_indices.h5'

        self.val_embed_path = 'C:\\w266\data2\\h5py_embeds\\squad_dev_embeds.h5'
        self.val_start_path = 'C:\\w266\data2\\h5py_embeds\\squad_dev_start_labels.h5'
        self.val_end_path = 'C:\\w266\data2\\h5py_embeds\\squad_dev_end_labels.h5'
        self.val_idx_path = 'C:\\w266\data2\\h5py_embeds\\squad_dev_indices.h5'

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
