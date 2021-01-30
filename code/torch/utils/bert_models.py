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
