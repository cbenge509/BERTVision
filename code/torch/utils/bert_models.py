import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

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

class MNLI_model(torch.nn.Module):
    '''Textual entailment BERT model'''
    def __init__(self, dropout_rate, hidden_state_size):
        super(MNLI_model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased').cuda()
        self.dropout = nn.Dropout(dropout_rate).cuda()
        self.cls = nn.Linear(hidden_state_size, 3).cuda()
        self.loss_fct = torch.nn.CrossEntropyLoss().cuda()

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        x = self.bert(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        x = x[1] #pull out cls token
        x = self.dropout(x)
        x = self.cls(x)
        self.logits = x
        self.loss = self.loss_fct(x, labels)
        return self


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

class MultiNNLayerParasiteLearned3(nn.Module):
    def __init__(self, hidden_state_size, freeze_bert = True):
        super(MultiNNLayerParasiteLearned3, self).__init__()
        self.nnlayer1 = NNLayer() #replace with BERTLayers
        self.nnlayer2 = NNLayer()
        self.nnlayer3 = NNLayer()
        self.nnlayer4 = NNLayer()

        if freeze_bert:
            for layer in dir(self):
                if 'nnlayer' in layer:
                    layer = getattr(self, layer)
                    for parameter in layer.parameters():
                        parameter.requires_grad = False

        self.p1 = Parasite(1, hidden_state_size)
        self.p2 = Parasite(2, hidden_state_size)
        self.p3 = Parasite(3, hidden_state_size)
        self.p4 = Parasite(4, hidden_state_size)

    def forward(self, x):
        x1 = self.nnlayer1(x)

        p1 = self.p1(x1.unsqueeze(1))
        x2 = self.nnlayer2(p1)

        p2 = self.p2(torch.stack([x1, x2], dim = 1))
        x3 = self.nnlayer3(p2+x1)

        p3 = self.p3(torch.stack([x1, x2, x3], dim = 1))
        x4 = self.nnlayer3(p3+p2+x1)

        p4 = F.relu(self.p4(torch.stack([x1, x2, x3, x4], dim = 1)))
        return p4

class Parasite(nn.Module):
    def __init__(self, hidden_states, token_size, bias_size):
        super(Parasite, self).__init__()
        self.params = torch.empty((hidden_states, token_size))
        nn.init.kaiming_normal_(self.params, mode='fan_out', nonlinearity='relu')
        self.params = torch.nn.Parameter(self.params)

        self.bias = torch.empty((1,bias_size))
        nn.init.kaiming_normal_(self.bias, mode='fan_out', nonlinearity='relu')
        self.bias = torch.nn.Parameter(self.bias.unsqueeze(0))

    def forward(self, x):
        #print(x.shape, self.params.shape, self.bias.shape)
        activations = torch.tensordot(x, self.params, dims=([1,2], [0,1])).unsqueeze(1)
        activations = activations + self.bias
        return torch.sigmoid(activations)

class MultiNNLayerParasiteLearnedBERT(nn.Module):
    def __init__(self, token_size = 100, bert_model = 'bert-base-uncased', freeze_bert = True):
        super(MultiNNLayerParasiteLearnedBERT, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

        if bert_model == 'bert-base-uncased':
            hidden_state_size = 768
        else:
            raise ValueError("Only bert-base-uncased is supported")

        #generate the BERT model
        self.bert_model = BertModel.from_pretrained(bert_model)

        #bert embeddings layer
        self.bert_embeddings = []

        #encoder layers
        self.encoder_layers = []

        for i,(name,module) in enumerate(self.bert_model.named_modules()):
            if i == 1:
                self.bert_embeddings.append(module)
            if 'encoder.layer.' in name and name[-1].isdigit():
                self.encoder_layers.append(module)

        #freeze BERT if desired
        if freeze_bert:
            for layer in self.bert_embeddings + self.encoder_layers:
                for parameter in layer.parameters():
                    parameter.requires_grad = False

        self.parasites = []
        for i in range(1, len(self.encoder_layers) + 1):
            p = Parasite(i, token_size, bias_size = hidden_state_size)
            setattr(self, "p%d"%i, p)
            self.parasites.append(p)

        self.output_layer = nn.Linear(768, 2)

    def forward(self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels = None
        ):
        """
        Take the same BERT inputs in order to process attention_mask
        """

        #get broadcastable attention mask for encoder layers
        extended_attention_mask = attention_mask[:, None, None, :]

        batch_size, seq_length = input_ids.size()

        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        #run through first embedding layer
        embeddings = self.bert_embeddings[0](input_ids=input_ids,
                                 token_type_ids=token_type_ids)

        prev_encoder_layers = [embeddings]
        for i in range(len(self.parasites)):
            p = self.parasites[i]
            encoder = self.encoder_layers[i]

            if i == 0:
                pi = p(embeddings.unsqueeze(1))
            else:
                pi = p(torch.stack(prev_encoder_layers, dim = 1))

            #add previous hidden states with learned parasite states
            x = self.encoder_layers[0](prev_encoder_layers[-1] + pi, attention_mask=extended_attention_mask)[0]
            prev_encoder_layers.append(x)
        #outputs
        pooled_output = x[:,0]
        pooled_output = self.output_layer(pooled_output)
        output = torch.sigmoid(pooled_output)
        #print(output, labels)
        #apply activation
        self.loss = self.criterion(output, labels).cuda()
        self.logits = output
        #print(self.loss)
        return self
