# packages
import sys
sys.path.append("C:/BERTVision/code/torch")
import torch, h5py


# prepare torch data set
class SSTH5Processor(torch.utils.data.Dataset):
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

    NAME = 'SSTH5'

    def __init__(self, type):
        self.type = type

        # todo: generalize locations
        self.train_embed_path = 'C:\\w266\data2\\h5py_embeds\\sst_bert_embeds.h5'
        self.train_label_path = 'C:\\w266\data2\\h5py_embeds\\sst_labels.h5'
        self.train_idx_path = 'C:\\w266\data2\\h5py_embeds\\sst_idx.h5'

        self.val_embed_path = 'C:\\w266\data2\\h5py_embeds\\sst_dev_bert_embeds.h5'
        self.val_label_path = 'C:\\w266\data2\\h5py_embeds\\sst_dev_labels.h5'
        self.val_idx_path = 'C:\\w266\data2\\h5py_embeds\\sst_idx.h5'

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
