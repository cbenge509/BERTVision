import h5py
import numpy as np
from utils.model_zoo import Models

batch_size = 32
data_size = 10

with h5py.File('./data/squad_train.h5', 'r') as train:
    start_ids = train['input_start']
    end_ids = train['input_end']
    labels = np.vstack([start_ids, end_ids]).T

indices = np.arange(len(labels))
np.random.shuffle(indices)
indices[:data_size]

model = Models('./models/').get_resnet50_v1_5()


#for i in range(100):
#    gen = BERTVision.utils.training_utils.simple_generator(labels, 
#                                                       '../data/train_bert_untuned_last_3/', 
#                                                       batch_size=batch_size,
#                                                       truncate_data = data_size,
#                                                       shuffle_index=indices)
#    history = model_full.fit(gen, steps_per_epoch=np.ceil(data_size/batch_size), epochs = 1)


#    def get_resnet50_v1_5(self, X = None, Y = None, batch_size = None, epoch_count = None, val_split = 0.1, shuffle = True,
#            recalculate_pickle = True, X_val = None, Y_val = None, task = "QnA", use_l2_regularizer = True,
#            batch_norm_decay = 0.9, batch_norm_epsilon = 1e-5, verbose = False, return_model_only = True):

print(indices)
