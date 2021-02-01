import torch
import torch.nn as nn
from transformers import BertModel

class LP_Model(torch.nn.Module):
    ''' Learned Pooling Model '''
    def __init__(self):
        super(LP_Model, self).__init__()
        self.n_layers = 13
        self.n_batch_sz = 16
        self.n_tokens = 384
        self.n_features = 768
        self.linear1 = torch.nn.Linear(1, 2)
        self.BC1 = BertConcat().cuda()

    def forward(self, x):
        x = self.BC1(x)  # [batch_sz, tokens, features, layers=1]
        x = self.linear1(x)  # [batch_sz, tokens, features, layers=2]
        start, end = torch.split(x, 1, dim=-1)
        start = torch.squeeze(start, dim=-1)  # [batch_sz, tokens, features]
        end = torch.squeeze(end, dim=-1)  # [batch_sz, tokens, features]
        return start, end

class RTE_model(torch.nn.Module):
    '''Textual entailment BERT model'''
    def __init__(self, dropout_rate, hidden_state_size):
        super(RTE_model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased').cuda()
        self.dropout = nn.Dropout(dropout_rate).cuda()
        self.cls = nn.Linear(hidden_state_size, 1).cuda()
        self.loss_fct = torch.nn.BCEWithLogitsLoss().cuda()

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        x = self.bert(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        x = x[1] #pull out cls token
        x = self.dropout(x)
        x = self.cls(x).squeeze(-1)
        self.logits = x
        self.loss = self.loss_fct(x, labels)
        return self

class MNLI(TwoSentenceLoader):
    NAME = 'MNLI'
    def __init__(self, type, transform = None):
        '''
        Line header:
        index	promptID	pairID	genre	sentence1_binary_parse	sentence2_binary_parse	sentence1_parse	sentence2_parse	sentence1	sentence2	label1	gold_label

        This prepares the RTE task from GLUE
        '''

        self.path = 'C:\w266\data\GLUE\MultiNLI (Matched and Mismatched)\MNLI'
        self.type = type
        if self.type == 'train':
            # initialize train
            self.train = pd.read_csv(self.path + '\\' + 'train.tsv', sep='\t',
                                     #names='id	qid1	qid2	question1	question2	is_duplicate'.split('\t'),
                                     encoding='latin-1',
                                     error_bad_lines=False,
                                     quoting = csv.QUOTE_NONE) #SOME BAD LINES IN THIS DATA

            self.train.columns = ['id', 'promptID', 'pairID', 'genre', 'sentence1_binary_parse', 'sentence2_binary_parse',
                                  'sentence1_parse', 'sentence2_parse', 'sentence1', 'sentence2', 'label1', 'gold_label']
            #Three labels: entailment neutral contradiction
            label_map = {'neutral':0,
                         'entailment':1,
                         'contradiction':2}
            self.train['label'] = [label_map[i] for i in self.train.gold_label]

        else:
            if self.type == 'dev_matched':
                # initialize dev (dev_matched set)
                self.dev = pd.read_csv(self.path + '\\' + 'dev_matched.tsv', sep='\t',
                                         #names='id	qid1	qid2	question1	question2	is_duplicate'.split('\t'),
                                         encoding='latin-1',
                                         error_bad_lines=False,
                                         quoting = csv.QUOTE_NONE)
            else:
                self.dev = pd.read_csv(self.path + '\\' + 'dev_mismatched.tsv', sep='\t',
                                         #names='id	qid1	qid2	question1	question2	is_duplicate'.split('\t'),
                                         encoding='latin-1',
                                         error_bad_lines=False,
                                         quoting = csv.QUOTE_NONE)

            self.dev.columns = ['id', 'promptID', 'pairID', 'genre', 'sentence1_binary_parse', 'sentence2_binary_parse',
                                  'sentence1_parse', 'sentence2_parse', 'sentence1', 'sentence2',
                                  'label1', 'label2', 'label3', 'label4', 'label5', 'gold_label']
            #Three labels: entailment neutral contradiction
            label_map = {'neutral':0,
                         'entailment':1,
                         'contradiction':2}
            self.dev['label'] = [label_map[i] for i in self.dev.gold_label]

        # initialize the transform if specified
        if transform:
            self.transform = transform
        else:
            self.transform = Tokenize_Transform()

    
class STSB_model(torch.nn.Module):
    '''Semantic Textual Similarity Benchmark (STS-B) Model
       ref: http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark'''

    def __init__(self, dropout_rate, hidden_state_size):
        super(STSB_model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased').cuda()
        self.dropout = nn.Dropout(dropout_rate).cuda()
        self.cls = nn.Linear(hidden_state_size, 1).cuda()
        self.loss_fct = torch.nn.MSELoss().cuda()

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        x = self.bert(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        x = x[1] #pull out cls token
        x = self.dropout(x)
        x = self.cls(x)
        self.logits = x
        self.loss = self.loss_fct(x, labels)
        return self
        x = self.cls(x).squeeze(-1)
        self.logits = x
        self.loss = self.loss_fct(x, labels).type(torch.float64)
        return self