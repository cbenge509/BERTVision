#%%
############################################################################
# IMPORTS
############################################################################
import os
import numpy as np
import h5py
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

#%%
############################################################################
# CLASS : BERTImageGenerator

#Example for obtaining data labels
#train = h5py.File('../SQuADv2/train_386.h5', 'r')
#start_ids = train['input_start']
#end_ids = train['input_end']
#labels = np.vstack([start_ids, end_ids]).T
############################################################################

class BERTImageGenerator(Sequence):
    """Keras data generator for reading .h5 files storing BERT images

    Args:
        data_dir ([str]): data directory containing images to load
        labels ([list]): ground-truth labels for the images
        batch_size ([int]): batch size of images to be returned 
        start_idx ([int]): starting index of the image generator
        end_idx ([int]): ending index of the image generator
        encoder_size ([int, optional]): encoder size for the BERT model, including output (13 for base, 24 for large)
        max_seq_length ([int, optional]): maximum query length for the model
        bert_embedding_dim ([int, optional]): BERT embedding dimension (786 for base, 1024 for large)
        include_output_seq ([bool, optional]): whether to include the output sequence in the training set
        shuffle ([bool, optional]): whether to shuffle data during training after each epoch
    Raises:
        RuntimeError: if data directory provided does not exist.
    """
    def __init__(self, 
                 data_dir,
                 labels, 
                 batch_size=32, 
                 start_idx = 0, 
                 end_idx = None, 
                 encoder_size = 25,
                 max_seq_length = 386,
                 bert_embedding_dim = 1024,
                 include_output_seq = False,
                 shuffle=True):
        
        if not os.path.isdir(data_dir):
            raise RuntimeError("Provided data directory does not exist")
            
        if end_idx is None:
            end_idx = len(labels)      
        
        self.shuffle = shuffle
        #keep track of data indices for generation
        self.indices = np.arange(start_idx, end_idx)
        self.labels = labels
        self.on_epoch_end()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.encoder_size = encoder_size
        self.max_seq_length = max_seq_length
        self.bert_embedding_dim = bert_embedding_dim
        self.include_output_seq = include_output_seq
        print('reloaded BERTImageGenerator')
        
    def __len__(self):
        '''Determines the number of batches per epoch'''
        return int(np.ceil((self.end_idx - self.start_idx) / self.batch_size))
    
    def __getitem__(self, idx):
        '''Retrieves the batch of examples'''
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        
        return self.__data_generator(indices)
        
    def __data_generator(self, indices):
        '''Generates a batch of data based on the sequences'''
        encoder_size = self.encoder_size
        if not self.include_output_seq:
            encoder_size -= 1
            
        X = np.empty((len(indices), encoder_size, self.max_seq_length, self.bert_embedding_dim))
        
        for i,idx in enumerate(indices):
            with h5py.File(self.data_dir + '/' + str(idx) + '.h5', 'r') as f_in:
                
                embedding = pad_sequences(f_in['hidden_state_activations'], 
                                          self.max_seq_length, 
                                          dtype = np.float32)
                
                if self.include_output_seq:
                    X[i] = embedding
                else:
                    X[i] = embedding[:-1]
        
        labels = self.labels[indices].T
        return X, [labels[0], labels[1]]
        #return np.swapaxes(X, 1, 3), [labels[0], labels[1]]
    
    def on_epoch_end(self):
        '''Performs data shuffling at the end of each epoch'''
        if self.shuffle == True:
            idx_shuffle = np.arange(len(self.indices), dtype = int)
            np.random.shuffle(idx_shuffle)
            self.indices = self.indices[idx_shuffle]

def simple_generator(labels, directory, batch_size = 32, truncate_data = False, shuffle_index = None):
    
    indices = np.arange(len(labels), dtype = int)
    size = len(indices)
    iterations = np.ceil(size / batch_size)
    if shuffle_index is None:
        np.random.shuffle(indices)
    else:
        indices = indices[shuffle_index]
    
    if truncate_data:
        indices = indices[:truncate_data]
        iterations = np.ceil(truncate_data / batch_size)
        
    darray = np.zeros((batch_size,386,1024,3), dtype = np.float32)
    
    offset = 0
    while True:
        darray[:] = 0
        data_slice = indices[offset * batch_size:(offset+1)*batch_size]
        for i,idx in enumerate(data_slice):
            with h5py.File(directory + '%d.h5' %(idx), 'r') as f:
                data = np.array(f['hidden_state_activations'], dtype = np.float32)
            darray[i, 386-data.shape[0]:386, :, :] = data
        
        #print('offset:', offset)
        #print('size:', len(data_slice))
        #print('data:', np.sum(darray))
        #print(len(data_slice))
        #print(batch_size)
        #print('\n', darray[0, -1, -1, -1], labels[data_slice][0], '\n')
        if len(data_slice) < batch_size:
            output = darray[:len(data_slice)]
            offset = 0
        else:
            offset += 1
            output = darray
        
        yield output, list(labels[data_slice].T)
        
def simple_dev_generator(labels, directory, batch_size = 32, truncate_data = False):
    indices = np.arange(len(labels), dtype = int)
    #np.random.shuffle(indices)
        
    darray = np.zeros((batch_size,386,1024,3), dtype = np.float32)
    
    offset = 0
    while True:
        data_slice = indices[offset * batch_size:(offset+1)*batch_size]
        for i,idx in enumerate(data_slice):
            with h5py.File(directory + '%d.h5' %(idx), 'r') as f:
                data = np.array(f['hidden_state_activations'], dtype = np.float32)
            darray[i, 386-data.shape[0]:386] = data
        #print(data_slice)
        if len(data_slice) < batch_size:
            output = darray[:len(data_slice)]
            offset = 0
        else:
            offset += 1
            output = darray
        yield output
