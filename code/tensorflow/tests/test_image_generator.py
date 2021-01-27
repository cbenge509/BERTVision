import unittest
import h5py

#git repo specific setup
import sys
sys.path.append('../utils')

from training_utils import *
from evaluation import *

class TestBERTImageGenerator(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = PretrainedBertSquad2Faster('../../bert_base_squad_1e-5_adam_4batchsize_2epochs_weights_BERT_ONLY.h5').model

        train = h5py.File('../../../SQuADv2/train_386.h5', 'r')
        self.start_ids = train['input_start']
        self.end_ids = train['input_end']
        self.train_inputs = np.array(train['input_ids'])
        self.train_masks = np.array(train['attention_mask'])
        self.labels = np.vstack([self.start_ids, self.end_ids]).T
        self.offset = 2
        self.start_idx = 14325
        self.end_idx = 14642

        self.images = BERTImageGenerator('../../../data/train/',
                            self.labels,
                            batch_size=1,
                            start_idx = self.start_idx,
                            end_idx = self.end_idx,
                            shuffle = False)

        self.subset = self.images[self.offset]

    def setUp(self):
        self.idx = self.start_idx + self.offset

    def test_embeddings_shape(self):
        #extract image 2 with starting index 14325, leading to image 14325

        self.assertEqual(len(self.subset), 2)
        self.assertEqual(self.subset[0].shape, (1, 24, 386, 1024))
        self.assertEqual(self.subset[1].shape, (1, 2))

    def test_offset(self):

        manual_load = h5py.File('../../../data/train/%d.h5' %self.idx, 'r')
        self.assertTrue((np.array(manual_load['hidden_state_activations'])[0][-1] == self.subset[0][0][0][-1]).all())

    def test_label_ids(self):
        self.assertEqual(self.subset[1][0][0], self.start_ids[self.idx])
        self.assertEqual(self.subset[1][0][1], self.end_ids[self.idx])

    def test_embeddings_with_model(self):

        embeddings, outputs = self.model.predict([self.train_inputs[[self.idx]], self.train_masks[[self.idx]]])
        non_zero_idx = np.sum(self.train_inputs[[self.idx]] != 0)
        self.assertTrue((embeddings[12][0][:non_zero_idx][-1] == self.subset[0][0][12][-1]).all())

    def test_shuffle_ids(self):
        pass

    def test_shuffled_labels(self):
        pass

if __name__ == '__main__':
    unittest.main()
