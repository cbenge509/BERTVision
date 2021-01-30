# packages
import sys
sys.path.append("C:/BERTVision/code/torch")
import torch, json, pytreebank
from transformers import BertTokenizerFast

class Tokenize_Transform():
    '''
    This function tokenize transforms the data organized in SSTProcessor().

    Parameters
    ----------
    sample : dict
        A dictionary containing: (1) text, (2) labels,
        and index positions

    Returns
    -------
    sample : dict
        A dictionary containing: (1) input tokens, (2) attention masks,
        (3) token type ids, (4) labels, and (5) data set index.
    '''
    def __init__(self):
        # instantiate the tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # retrieve sample and unpack it
    def __call__(self, sample):
        # transform text to input ids and attn masks

        if 'text2' not in sample:
            encodings = self.tokenizer(
                                sample['text'],  # document to encode.
                                add_special_tokens=True,  # add '[CLS]' and '[SEP]'
                                max_length=64,  # set max length; SST is 64
                                truncation=True,  # truncate longer messages
                                padding='max_length',  # add padding
                                return_attention_mask=True,  # create attn. masks
                                return_tensors='pt'  # return pytorch tensors
                           )
        else:
            encodings = self.tokenizer(
                                sample['text'],  # document to encode.
                                sample['text2'], #second sentence to encode
                                add_special_tokens=True,  # add '[CLS]' and '[SEP]'
                                max_length=512,  # set max length; SST is 64
                                truncation=True,  # truncate longer messages
                                padding='max_length',  # add padding
                                return_attention_mask=True,  # create attn. masks
                                return_tensors='pt'  # return pytorch tensors
                           )

        # package up encodings
        return {'input_ids': torch.as_tensor(encodings['input_ids'],
                                         dtype=torch.long),

                'attention_mask': torch.as_tensor(encodings['attention_mask'],
                                          dtype=torch.long),

                 'token_type_ids': torch.as_tensor(encodings['token_type_ids'],
                                                   dtype=torch.long),

                'labels': torch.as_tensor(sample['label'],
                                          dtype=torch.long),

                'idx': torch.as_tensor(sample['idx'],
                                       dtype=torch.int)}
