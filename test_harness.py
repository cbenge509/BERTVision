import h5py
import numpy as np
from utils.model_zoo import Models
zoo = Models('./models/')

batch_size = 32

# load the labels from the SQuAD v2 Features
with h5py.File('./data/squad_train.h5', 'r') as train_data:
    train_ids = np.array(train_data['input_ids'], dtype = np.int32)
    train_masks = np.array(train_data['attention_mask'], dtype = np.int32)
    train_input_start = np.array(train_data['start_position'], dtype = np.int32)
    train_input_end = np.array(train_data['end_position'], dtype = np.int32)

Y = np.vstack([train_input_start, train_input_end]).T[0:batch_size] # only the first 32 images / labels

# Load some of the images (only 32 on my local drive)
X = np.zeros((batch_size, 386, 1024, 3), dtype = np.float32)
for i, idx in enumerate(np.arange(0, len(Y))):
    with h5py.File('./data/train_bert_untuned_first_3/%d.h5' % (idx), 'r') as f:
        data = np.array(f['hidden_state_activations'], dtype = np.float32)[0:386]
        # center the 386 dim
        X[i, (386 - data.shape[0])//2 + (386 - data.shape[0]) % 2:386 - (386 - data.shape[0])//2, :, :] = data

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