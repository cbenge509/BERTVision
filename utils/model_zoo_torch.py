import torch
from torch import nn
from scipy.stats import truncnorm

class BertConcat(nn.Module):
    def __init__(self, seq_length, embeddings, ha, bias=True):
        super().__init__()
        
        #Will only work currently with units = 1
        self.units = 1

        self.w = nn.Parameter(torch.nn.init.normal_(torch.Tensor(ha), mean = 0, std = 0.05))
        self.t = nn.Parameter(torch.Tensor(1))

    def forward(self, inputs):
        sm = nn.Softmax()
        w = sm(self.w)
        return torch.sum(torch.multiply(inputs, w), dim = -1, keepdims = True) * self.t


def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.
    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + torch.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))))
    return x * cdf


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

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


#### Not finished
class AdapterPooler(nn.Module):
    def __init__(self, input_dim, adapter_dim, init_scale = 1e-3, shared_weights = True):
        super().__init__()
        self.adapter_dim = adapter_dim
        if shared_weights:
            self.pooler_layer = TimeDistributed(
                torch.nn(input_dim, adapter_dim))
        else:
            raise NotImplementedError
            ##??self.pooler_layer = tf.keras.layers.LocallyConnected1D(self.adapter_dim, 1, 1, kernel_initializer=self.initializer)

    def forward(self, inputs):
        '''Input shape expected to be (batch_size, 386, 1024, 24)
           Call reshapes tensor into (batch_size * 386, 24, 1024)
           Apply pooler_layer to input with gelu activation
        '''
        sequence_dim = inputs.shape[1]
        embedding_dim = inputs.shape[2]
        encoder_dim = inputs.shape[3]

        #Combine batch and sequence length dimension
        X = torch.reshape(inputs, [-1, embedding_dim, encoder_dim])

        #Move encoder_dim to axis = 1
        X = torch.transpose(X, (0, 2, 1))

        X = self.pooler_layer(X)
        X = gelu(X)

        #Regenerate shape
        X = torch.transpose(X, (0, 2, 1))
        X = torch.reshape(X, [-1, sequence_dim, self.adapter_dim, encoder_dim])

        return X
