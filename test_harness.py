############################################################################
## IMPORTS
############################################################################
import h5py
import numpy as np
import os
from utils.model_zoo import Models
from utils.squad import FineTunedBertSquad2
from progressbar import ProgressBar
zoo = Models('./models/')

############################################################################
## CONSTANTS
############################################################################

TEST_RESNET50_V1_5 = False
TEST_XCEPTION = False
TEST_SMALL = False
TEST_TENNEY = True

#TASK = "binary_classification"
TASK = "QnA"
VERBOSE = True
BATCH_SIZE = 64
EPOCHS = 10
VAL_SPLIT = 0.1
#TRAIN_DATA_FOLDER = 'C:/MIDS/W266 - Natural Language Processing/Repos/BERTVision/data/train_bert_untuned_last_3_full386_tokenids_specified/'
#TRAIN_LABEL_FILE = 'C:/MIDS/W266 - Natural Language Processing/Repos/BERTVision/data/squad_train.h5'
TRAIN_DATA_FOLDER = 'C:/w266/data/train_bert_untuned_last_3_full386_tokenids_specified/'
TRAIN_LABEL_FILE = 'C:/w266/stone/BERTVision/data/squad_train.h5'
BERT_WEIGHTS_FILE = 'C:/w266/stone/bert_base_squad_1e-5_adam_4batchsize_4epochs_weights_BERT_ONLY.h5'

#Toggle to go between generating data (for small amounts) and loading data
LOAD_MODE = 0

#TRAIN_SIZE = 25000
TRAIN_SIZE = 2400
DEV_SIZE = 200

############################################################################
# UTILITIES
############################################################################

def get_file_indices(data_folder):

    #return np.sort([int(f.replace('.h5', '')) for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f)) & f.endswith('.h5')])
    return np.arange(2500)

def load_labels(label_file, qna = True):

    with h5py.File(label_file, 'r') as train:
        start_ids = train['start_position']
        end_ids = train['end_position']
        labels = np.vstack([start_ids, end_ids]).T

    if qna:
        return labels
    else:
        return np.array([1 if sum(i) > 0 else 0 for i in labels])

# NOTE: right now, we are using TRAIN data and labels for dev, just the tail end
def load_dev(dev_size, data_folder, dev_labels, dev_indices, qna = True, verbose = False):


    dev = np.zeros((dev_size, 386, 1024, 3), dtype = np.float32)

    for i, j in enumerate(dev_indices[-dev_size:]):
        with h5py.File(os.path.join(data_folder, f"{j}.h5"), 'r') as f:
            dev[i] = np.array(f['hidden_state_activations'])
        if verbose:
            if not i % 100: print(i)

    if qna:
        return dev, tuple(dev_labels[dev_indices[-dev_size:]].T)
    else:
        return dev, dev_labels[dev_indices[-dev_size:]].astype(np.uint8)

def load_train(train_index, train_indices, train_size, data_folder, train_labels, qna = True, verbose = False):

    index = train_indices[train_index * train_size:(train_index + 1) * train_size]
    train = np.zeros((train_size, 386, 1024, 3), dtype = np.float32)
    labels = [np.zeros(train_size), np.zeros(train_size)]

    for i, j in enumerate(index):
        with h5py.File(os.path.join(data_folder, f"{j}.h5"), 'r') as f:
            train[i] = np.array(f['hidden_state_activations'])
        if verbose:
            if not i % 100: print(i)

    labels[0], labels[1] = train_labels[index].T[0], train_labels[index].T[1]

    if qna:
        return train, np.array(labels)
    else:
        return train, train_labels[index].astype(np.uint8)

def generate_data(bert_weights,
                  train_size, dev_size,
                  train_squad):
    try:
        train_embeddings = np.zeros((train_size, 386, 1024, 24))
        dev_embeddings = np.zeros((dev_size, 386, 1024, 24))

    except MemoryError:
        print("Out of memory: try reducing training/dev size")

    model = FineTunedBertSquad2(bert_weights)

    with h5py.File(train_squad, 'r') as train:
        start_ids = train['start_position']
        end_ids = train['end_position']
        labels = np.vstack([start_ids, end_ids]).T

        train_ids = np.array(train['input_ids'], dtype = np.int32)
        train_masks = np.array(train['attention_mask'], dtype = np.int32)
        train_tokens = np.array(train['token_type_ids'], dtype = np.int32)

    print("Getting data...grab a drink and enjoy the animation")
    pbar = ProgressBar().start()
    for i in range(train_size):
        e = model.model.predict([train_ids[[i]], train_masks[[i]], train_tokens[[i]]])
        for j in range(24):
            train_embeddings[i, :, :, j] = np.squeeze(e[j], axis = 0)
        pbar.update(i + 1)

    for i in range(train_size, train_size + dev_size):
        e = model.model.predict([train_ids[[i]], train_masks[[i]], train_tokens[[i]]])
        for j in range(24):
            dev_embeddings[i - train_size, :, :, j] = np.squeeze(e[j], axis = 0)

        pbar.update(i + 1)
    pbar.finish()
    train_labels = list(labels[:train_size].T)
    dev_labels = tuple(labels[train_size:train_size+dev_size].T)

    return train_embeddings, train_labels, dev_embeddings, dev_labels

############################################################################
# MAIN FUNCTION
############################################################################

if __name__ == "__main__":

    # Clear the screen
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')

    if LOAD_MODE:
        if VERBOSE: print("Loading TRAIN file indices...")
        train_indices = get_file_indices(TRAIN_DATA_FOLDER)
        if VERBOSE: print("Loading TRAIN labels...")
        train_labels = load_labels(label_file = TRAIN_LABEL_FILE, qna = TASK == "QnA")

        if VERBOSE: print("Loading DEV (Validation) examples & labels...")
        # NOTE: right now, we are using the last {DEV_SIZE} files in the TRAIN set as our DEV (Validation) set
        # this will change in the future to the actual dev set once we get the utiility set up for that.
        dev, dev_labels = load_dev(dev_size = DEV_SIZE, data_folder = TRAIN_DATA_FOLDER, dev_labels = train_labels,
            dev_indices = train_indices, qna = TASK == "QnA", verbose = VERBOSE)


    if TEST_SMALL:

#            def get_small(self, X = None, Y = None, batch_size = None, epoch_count = 10, val_split = 0.1, shuffle = True, input_shape = (386, 1024, 3),
#            recalculate_pickle = True, X_val = None, Y_val = None, task = "QnA", verbose = False, return_model_only = True):

        if VERBOSE:print("Loading TRAIN data & labels...")
        X, Y = load_train(train_index = 0, train_indices = train_indices, train_size = TRAIN_SIZE,
            data_folder = TRAIN_DATA_FOLDER, train_labels = train_labels, qna = TASK == "QnA", verbose = VERBOSE)

        if VERBOSE: print("\n\Training Small...\n")
        model, hist_params, hist = zoo.get_small(X = X, Y = Y, batch_size = BATCH_SIZE, epoch_count = EPOCHS, X_val = dev, Y_val = dev_labels,
            shuffle = True, input_shape = (386, 1024, 3), recalculate_pickle = True, task = TASK, return_model_only = False, verbose = VERBOSE)

        print("\n", hist)


    if TEST_XCEPTION:

        if VERBOSE:print("Loading TRAIN data & labels...")
        X, Y = load_train(train_index = 0, train_indices = train_indices, train_size = TRAIN_SIZE,
            data_folder = TRAIN_DATA_FOLDER, train_labels = train_labels, qna = TASK == "QnA", verbose = VERBOSE)

        if VERBOSE: print("\n\Training Xception...\n")
        model, hist_params, hist = zoo.get_xception(X = X, Y = Y, batch_size = BATCH_SIZE, epoch_count = EPOCHS, X_val = dev, Y_val = dev_labels,
            shuffle = True, input_shape = (386, 1024, 3), recalculate_pickle = True, task = TASK, use_jiang_reduction = True,
            return_model_only = False, verbose = VERBOSE)

#        model, hist_params, hist = zoo.get_xception(X = X, Y = Y, batch_size = BATCH_SIZE, epoch_count = EPOCHS, val_split = VAL_SPLIT,
#            shuffle = True, input_shape = (386, 1024, 3), recalculate_pickle = True, task = "QnA", use_jiang_reduction = False,
#            return_model_only = False, verbose = VERBOSE)

        #if VERBOSE: print("Reloading Xception from pickle file...")
        #model, hist_params, hist = zoo.get_xception(X = None, Y = None, batch_size = None, epoch_count = None, recalculate_pickle = False)



    # NOTE: THIS IS BROKEN, DO NOT RUN FOR NOW (CDB: 6/29/2020)
    if TEST_RESNET50_V1_5:
        # TODO: this one needs to be reworked; was implemented before the (above) loading scheme was implemented.
        print("Evaluating ResNet50 v1.5...")
        # TEST : Train model
        model, hist_params, hist = zoo.get_resnet50_v1_5(X = X, Y = Y, batch_size = batch_size, epoch_count = 50, val_split = 0.1,
            shuffle = True, recalculate_pickle = True, task = "QnA", use_l2_regularizer = True, batch_norm_decay = 0.9,
            batch_norm_epsilon = 1e-5, verbose = True, return_model_only = False)

        # TEST : Retrieve model from pickle
        model, hist_params, hist = zoo.get_resnet50_v1_5(X = None, Y = None, batch_size = None, epoch_count = None, recalculate_pickle = False)
        print(model.summary())

        # TEST : predict values and evaluate results
        pred_start, pred_end = zoo.predict_resnet50_v1_5(X = X, verbose = True)
        print(pred_start, pred_end)
        print(pred_start.shape, pred_end.shape)

    if TEST_TENNEY:
        X, Y, X_val, Y_val = generate_data(BERT_WEIGHTS_FILE, TRAIN_SIZE, DEV_SIZE, TRAIN_LABEL_FILE)
        model = zoo.baseline_tenney_weighting((386,1024,24))

        print(model.summary())
        model.fit(X, Y,
                  validation_data = (X_val, Y_val),
                  batch_size = BATCH_SIZE,
                  epochs = 10,
                  shuffle = False)
