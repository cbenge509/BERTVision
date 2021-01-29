import IPython
import unittest
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizerFast
from processors import QQPairs

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
def qqpairs_train_20():
    pairs = pd.read_csv('C:\w266\data\GLUE\Quora Question Pairs\QQP\\train.tsv', encoding='latin-1', sep='\t')
    pairs_20 = pairs.iloc[19,:]
    text = tokenizer(pairs_20.question1, pairs_20.question2,
                     add_special_tokens=True,  # add '[CLS]' and '[SEP]'
                     max_length=512,  # set max length; SST is 64
                     truncation=True,  # truncate longer messages
                     padding='max_length',  # add padding
                     return_attention_mask=True,  # create attn. masks
                     )

    return np.array(text['input_ids']), np.array(text['token_type_ids']), np.array(text['attention_mask'])

class TestQQPairs(unittest.TestCase):

    def test_example_20(self):
        pairs = QQPairs('train')
        pairs_example_20 = pairs[19]


        truth_input, truth_type, truth_attention = qqpairs_train_20()
        self.assertTrue(np.all(pairs_example_20['input_ids'].numpy() == truth_input))
        self.assertTrue(np.all(pairs_example_20['token_type_ids'].numpy() == truth_type))
        self.assertTrue(np.all(pairs_example_20['attention_mask'].numpy() == truth_attention))


if __name__ == '__main__':
    unittest.main()
