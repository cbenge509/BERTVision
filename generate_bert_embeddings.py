############################################################################
# IMPORTS
############################################################################

import numpy as np
from utils import squad
from itertools import product
import argparse
import os
import h5py
from utils.squad import UntrainedBertSquad2
############################################################################
# ARGUMENT SPECIFICATION
############################################################################
def dir_is_writable_path(path):
    if os.access(path, os.W_OK):
        return path
    else:
        raise ValueError("Provided directory does not exist or is not writable")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-features', required=True, type=argparse.FileType('r'), help="Input .h5 squad features train file")
    parser.add_argument('--dev-features', required=True, type=argparse.FileType('r'), help="Input .h5 squad features dev file")
    parser.add_argument('-o', '--output-dir', required=True, type=dir_is_writable_path, help="Specify the output directory for the train embeddings")
    parser.add_argument('--model', default = UntrainedBertSquad2().model, help="Model object that can run predict on squad features")
    return parser.parse_args()

############################################################################
# PROCESS SQUAD V2 DATA
############################################################################
def get_outputs(embeddings, not_pad, outputs):
    embeddings = np.transpose(embeddings, axes=( 2, 3, 0, 1)).squeeze(axis = 3)
    outputs = np.transpose(outputs, axes = (1,2,0)).squeeze(axis = 2)
    return embeddings[not_pad, :, :], outputs[not_pad, :]

def write_file(directory, idx, embeddings):
    with h5py.File(os.path.join(directory, str(idx) + '.h5'), 'w') as f:
        f.create_dataset('hidden_state_activations', data = embeddings)

def write_sequence_outputs(directory, idx, outputs):
    with h5py.File(os.path.join(directory, str(idx) +'.h5'), 'w') as f:
        f.create_dataset('sequence_outputs', data = outputs)

def gen_directory_structure(directory):
    data_types = ['train', 'dev']
    seq_types = ['first_3', 'last_3', 'sequence_outputs']

    dirs = list(product(data_types, seq_types))
    output_dirs = [os.path.join(directory, *d) for d in dirs]

    for output_dir in output_dirs:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    return output_dirs

def gen_embeddings(model,
                   train_features,
                   dev_features,
                   output):

    train_first, train_last, train_seq, dev_first, dev_last, dev_seq = gen_directory_structure(output)
    with h5py.File(train_features, 'r') as train_data:
        train_ids = np.array(train_data['input_ids'], dtype = np.int32)
        train_masks = np.array(train_data['attention_mask'], dtype = np.int32)
        train_tokens = np.array(train_data['token_type_ids'], dtype = np.int32)

    with h5py.File(dev_features, 'r') as dev_data:
        dev_ids = np.array(dev_data['input_ids'], dtype = np.int32)
        dev_masks = np.array(dev_data['attention_mask'], dtype = np.int32)
        dev_tokens = np.array(dev_data['token_type_ids'], dtype = np.int32)

    #training data
    for i in range(len(train_ids)):
        embeddings, outputs = model.predict([train_ids[[i]], train_masks[[i]], train_tokens[[i]]])
        not_pad = np.where(train_ids[i] != 0)[0]
        e, o = get_outputs(embeddings, not_pad, outputs)
        write_file(train_first, i, e[:,:,:3])
        write_file(train_last, i, e[:,:,-4:-1])
        write_sequence_outputs(train_seq, i, o)

    #dev data
    for i in range(len(dev_ids)):
        embeddings, outputs = model.predict([dev_ids[[i]], dev_masks[[i]], dev_tokens[[i]]])
        not_pad = np.where(dev_ids[i] != 0)[0]
        e, o = get_outputs(embeddings, not_pad, outputs)
        write_file(dev_first, i, e[:,:,:3])
        write_file(dev_last, i, e[:,:,-4:-1])
        write_sequence_outputs(dev_seq, i, o)

############################################################################
# MAIN FUNCTION
############################################################################

if __name__ == '__main__':
    args = parse_args()
    gen_embeddings(args.model,
                   args.train_features,
                   args.dev_features,
                   args.output_dir)
