import torch
import torch.nn as nn
# with collate function, H5 data is emitted as:
# h5 emits [layers, batch_sz, tokens, features]



class BertConcat(torch.nn.Module):
    '''Implementation of learned pooler reported by Tenney 2019
       Original paper: https://arxiv.org/abs/1905.05950
    '''
    def __init__(self, n_layers):
        super(BertConcat, self).__init__()
        self.n_layers = n_layers
        # weight shape: [num_layers,]
        self.weights = torch.zeros(self.n_layers, dtype=torch.float32).normal_(0.0, 0.1).cuda()
        self.bias = torch.ones(1, dtype=torch.float32).cuda()

    def forward(self, x):
        '''
        Parameters
        ----------
        embeddings : torch tensor
            An embedding tensor with shape: [layers, batch_sz, tokens, features]

        Returns
        -------
        x  : torch tensor
            A tensor containing reduced embeddings with the following
            dimensions: [batch_sz, tokens, features, layers = 1]
        '''
        # expects data shaped: [batch_sz, tokens, features, layers]
        # so change shape to:
        x = x.permute(1, 2, 3, 0) # [batch_sz, tokens, features, layers]
        x = torch.softmax(x, dim=-1)  # last dim
        x = torch.sum(torch.mul(x, self.weights),
                               dim=-1, keepdims=True) * self.bias
        return x


class TimeDistributed(nn.Module):
    """ Takes an operation and applies it for each time sample
   Ref: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
   Args:
      module (nn.Module): The operation
      batch_first (bool): If true, x is (samples, timesteps, output_size)
         Else, x is (timesteps, samples, output_size)
   """

    def __init__(self, n_tokens, n_features, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.n_tokens = n_tokens
        self.n_features = n_features
        self.module =  torch.nn.Linear(self.n_features, self.n_tokens)
        self.batch_first = batch_first

    def forward(self, x):
        # expects tensor shaped: [batch_sz, features, tokens]
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y



class AdapterPooler(torch.nn.Module):
    '''
    Pooler inspired by Adapter BERT
    Original paper: https://arxiv.org/abs/1902.00751
    GitHub: https://github.com/google-research/adapter-bert

    Parameters
    ----------
    shared_weights : flag
        Whether to use the TimeDistributed module or a CNN (not implemented)

    embeddigns     : torch tensor [layers, batch_sz, tokens, features]

    Returns
    -------
    embeddings  : torch tensor
        A tensor containing embeddings with the following
        dimensions: [batch_sz, tokens, tokens, layers]
    '''
    def __init__(self, layers, batch_sz, tokens, features, shared_weights=True):
        super(AdapterPooler, self).__init__()
        self.GELU = torch.nn.GELU()
        self.n_layers = layers
        self.n_batch_sz = batch_sz
        self.n_tokens = tokens
        self.n_features = features
        if shared_weights:
            self.pooler_layer = TimeDistributed(self.n_tokens, self.n_features)
        else:
            raise NotImplementedError

    def forward(self, x):
        # h5 emits [layers, batch_sz, tokens, features], so change to expected:
        x = x.permute(1, 2, 3, 0)  # [batch_sz, tokens, features, layers]
        # reshape
        x = x.reshape(-1, self.n_features, self.n_layers)  # [tokens*batch_sz, features, layers]
        # move axis
        x = x.permute(0, 2, 1)  # [tokens*batch_sz, layers, features]
        # apply pooler_layer
        x = self.pooler_layer(x) # [tokens*batch_sz, layers, tokens]
        # apply GELU
        x = self.GELU(x)
        # move axis
        x = x.permute(0, 2, 1)  # [tokens*batch_sz, tokens, layers]
        # reshape
        x = torch.reshape(x, shape=(-1, self.n_tokens, self.n_tokens, self.n_layers))
        return x  # [batch_sz, tokens, tokens, layers]



class AP_SQuAD(torch.nn.Module):
    ''' Adaptive Pooler Model for SQuAD '''
    def __init__(self, n_layers, n_batch_sz, n_tokens, n_features):
        super(AP_SQuAD, self).__init__()
        self.n_layers = n_layers
        self.n_batch_sz = n_batch_sz
        self.n_tokens = n_tokens
        self.n_features = n_features
        self.linear1 = torch.nn.Linear(self.n_tokens*self.n_layers, 2)
        self.linear2 = torch.nn.Linear(self.n_tokens, 2)
        self.AP1 = AdapterPooler(self.n_layers, self.n_batch_sz, self.n_tokens, self.n_features)

    def forward(self, x):
        x = self.AP1(x)  # yields [batch_sz, tokens, tokens, layers]
        # reshape to [batch_sz, tokens, tokens*layers]
        x = torch.reshape(x, shape=(-1, self.n_tokens, self.n_tokens*self.n_layers))
        # output layer
        x = self.linear1(x)  # [batch_sz, tokens, layers=2]
        # split on last dim
        start, end = torch.split(x, 1, dim=-1)
        start = torch.squeeze(start, dim=-1)  # [batch_sz, tokens]
        end = torch.squeeze(end, dim=-1)  # [batch_sz, tokens]
        return start, end



class AP_GLUE(torch.nn.Module):
    '''
        Adapter Pooler for GLUE
        Data emitted from H5 Processor is shape:
        [batch_sz, layers, tokens, features]
    '''
    def __init__(self, n_layers, n_batch_sz, n_tokens, n_features, n_labels):
        super(AP_GLUE, self).__init__()
        self.n_layers = n_layers
        self.n_batch_sz = n_batch_sz
        self.n_tokens = n_tokens
        self.n_features = n_features
        self.num_labels = n_labels
        self.AP1 = AdapterPooler(self.n_layers, self.n_batch_sz, self.n_tokens, self.n_features)
        self.linear1 = torch.nn.Linear(self.n_tokens*self.n_tokens*self.n_layers, self.num_labels)
        return None

    def forward(self, x):
        x = self.AP1(x)  # yields [batch_sz, tokens, tokens, layers]
        # reshape to [batch_sz, tokens, tokens*layers]
        x = torch.reshape(x, shape=(-1, self.n_tokens, self.n_tokens*self.n_layers))
        # reshape again to combine dimensions
        x = x.view(-1, self.n_tokens*self.n_tokens*self.n_layers)  # [batch_sz, tokens*tokens*layers]
        # output layer
        x = self.linear1(x)  # [batch_sz, num_labels]
        return x

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * x.sigmoid()

class DetectParts(nn.Module):
    """
    d_depth = n  # number of layers (embeddings per token)
    d_emb = n  # dimension of token embeddings (features)
    d_inp = n  # number of features computed per embedding
    """
    def __init__(self, d_depth, d_emb, d_inp):
        super().__init__()
        self.depth_emb = nn.Parameter(torch.zeros(d_depth, d_emb))
        self.detect_parts = nn.Sequential(nn.Linear(d_emb, d_inp), Swish(), nn.LayerNorm(d_inp))
        nn.init.kaiming_normal_(self.detect_parts[0].weight)
        nn.init.zeros_(self.detect_parts[0].bias)
        # bert emits ([13, 16, 512, 768])
    def forward(self, x):
        x = x.permute(1, 2, 0, 3)                          # [bs, tokens, layers=1, features]
        x = self.detect_parts(x + self.depth_emb)          # [bs, n, d_depth, d_inp]
        x = x.view(x.shape[0], -1, 1, x.shape[-1])         # [bs, (n * d_depth), 1, d_inp]
        # outputs [bs, tokens, features]
        return x.squeeze(2)



class DP_Model(torch.nn.Module):
    '''
    Detect Parts Demo
    '''
    def __init__(self, layers, batch_sz, tokens, features, labels):
        super(DP_Model, self).__init__()
        self.GELU = torch.nn.GELU()
        self.n_layers = layers
        self.n_batch_sz = batch_sz
        self.n_tokens = tokens
        self.n_features = features
        self.dropout = torch.nn.Dropout(0.3)
        self.n_labels = labels
        self.d_depth = 13
        self.d_emb = 768
        self.d_inp = 128
        self.DP = DetectParts(d_depth=self.d_depth, d_emb=self.d_emb, d_inp=self.d_inp)
        self.out_layer = torch.nn.Linear(self.d_inp*self.n_tokens*self.d_depth, self.n_labels)

    def forward(self, x):
        # BERT/H5 emits: # [layers, batch_sz, tokens, features]]
        x = self.DP(x)
        x = x.view(-1, self.n_layers*self.n_tokens*self.d_inp)
        x = self.GELU(x)
        x = self.dropout(x)
        x = self.out_layer(x)
        return x


#
