###########################################################################################################
## IMPORTS
###########################################################################################################
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import tensorflow as tf

from tensorflow.keras.layers import LeakyReLU, ELU, ReLU
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Activation, Convolution2D, Conv2D, LocallyConnected2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, SeparableConv2D
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense, Dropout, Input, concatenate, add, Add, ZeroPadding2D, GlobalMaxPooling2D, DepthwiseConv2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import metrics, losses, initializers, backend
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.initializers import glorot_uniform, Constant, lecun_uniform
from tensorflow.keras import backend as K

from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split

from tensorflow.python.keras import backend, initializers, models, regularizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
layers = tf.keras.layers
np.random.seed(42)
tf.random.set_seed(42)
tf.get_logger().setLevel('ERROR')
physical_devices = tf.config.list_physical_devices('GPU')
for pd_dev in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[pd_dev], True)


###########################################################################################################
## HELPER FUNCTIONS
###########################################################################################################

def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      x: float Tensor to perform activation.
    Returns:
      `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))

    return x * cdf

###########################################################################################################
## CUSTOM TENSORFLOW LAYERS
###########################################################################################################

class BertConcat(layers.Layer):
    def __init__(self, units = 1):
        super().__init__()

        #Will only work currently with units = 1
        self.units = 1

    def build(self, input_shape):
        self.w = self.add_weight(shape = (input_shape[-1],), trainable = True, initializer = 'random_normal', name = 'weights')
        self.t = self.add_weight(shape = (1), trainable = True, initializer = 'ones', name = 'probes')

    def call(self, inputs):
        w = tf.nn.softmax(self.w)
        return tf.tensordot(w, inputs, axes = (0, -1)) * self.t

class AdapterPooler(tf.keras.layers.Layer):
    def __init__(self, adapter_dim, init_scale = 1e-3, shared_weights = True):
        super().__init__()
        self.adapter_dim = adapter_dim
        self.initializer = tf.keras.initializers.TruncatedNormal(stddev=init_scale)
        if shared_weights:
            self.pooler_layer = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self.adapter_dim, kernel_initializer=self.initializer))
        else:
            self.pooler_layer = tf.keras.layers.LocallyConnected1D(self.adapter_dim, 1, 1, kernel_initializer=self.initializer)

    def call(self, inputs):
        '''Input shape expected to be (batch_size, 386, 1024, 24)
           Call reshapes tensor into (batch_size * 386, 24, 1024)
           Apply pooler_layer to input with gelu activation
        '''

        sequence_dim = inputs.shape[1]
        embedding_dim = inputs.shape[2]
        encoder_dim = inputs.shape[3]

        #Combine batch and sequence length dimension
        X = tf.reshape(inputs, [-1, embedding_dim, encoder_dim])

        #Move encoder_dim to axis = 1
        X = tf.transpose(X, (0, 2, 1))

        X = self.pooler_layer(X)
        X = gelu(X)

        #Regenerate shape
        X = tf.transpose(X, (0, 2, 1))
        X = tf.reshape(X, [-1, sequence_dim, self.adapter_dim, encoder_dim])

        return X

class MeanConcat(layers.Layer):
    def __init__(self, units = 1):
        super().__init__()

        #Will only work currently with units = 1
        self.units = 1

    def build(self, input_shape):
        self.last_axis = len(input_shape) - 1

    def call(self, inputs):
        return tf.reduce_mean(inputs, self.last_axis)

###########################################################################################################
## CLASS CONTAINING BINARY CLASSIFICATION MODELS
###########################################################################################################
class BinaryClassificationModels(object):

    def __init__(self, **kwargs):
        self.__GPU_count = len(tf.config.list_physical_devices('GPU'))

    ######################################################
    ### Private Methods
    ######################################################

    # validate required input parameter values aren't set to None
    @staticmethod
    def __require_params(**kwargs):
        
        needed_args = [key for key,value in kwargs.items() if value is None]
        if len(needed_args) > 0:
            raise ValueError("If running in training, must specify following outputs: %s" %(', '.join(needed_args)))
    
        return

    def __verbose_print(self, model, model_name, input_shape, opt, loss, metrics):

        print("".join(["\n", "*" * 100, "\nModel Details\n", "*" * 100, "\n"]))
        print(f"Model Name: {model_name}")
        print(f"Optimizer Details:  name = {opt.get_config()['name']},  learning rate = {opt.get_config()['learning_rate']}")
        print(f"Loss Details:  name = {loss.get_config()['name']}, from_logits = {loss.get_config()['from_logits']}")
        print(f"Input Shape: {tuple(input_shape)}")
        print(f"Metrics: {"".join(metrics)}")
        print("*" * 100)
        print(model.summary())
        print("*" * 100, "\n")

        return

    ######################################################
    ### Public Methods
    ######################################################

    #/////////////////////////////////////////////////////
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    ### "Tiny" Tenney
    #/////////////////////////////////////////////////////
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    def get_tiny_tenney(self, input_shape = (1, 1024, 26), gpu_device = "/gpu:0", verbose = True):
        r"""Returns the TensorFlow 2.2 implementation of 'Tiny' Tenny linear model.
            Inspired by Tenney et al. 2019 : https://arxiv.org/abs/1905.05950

        Args:
            input_shape (tuple, optional): Shape of the input tensor. Defaults to (1, 1024, 26).
            gpu_device (str, optional): If GPU devices are available, defines which one to utilize. Defaults to "/gpu:0".
            verbose (bool, optional): Log details to console. Defaults to True.

        Returns:
           tf.keras.Model: returns a TensorFlow 2.20 model (compiled, untrained)
        """

        # Input validation
        self.__require_params(input_shape = input_shape)

        if (not gpu_device) or self.__GPU_count == 0:
            gpu_device = "/cpu:0"
        
        # Model hyperparameters and metadata
        model_name = 'Binary Classification Tenney (Tiny)'
        opt = Adam(lr = 1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
        loss = CategoricalCrossentropy(from_logits = True)
        metrics = ['accuracy']

        # Construct model & compile
        with tf.device(gpu_device):
            inp = layers.Input(input_shape, name = 'input_layer')
            X = BertConcat() (inp)
            X = tf.squeeze(X, axis = 1)
            X = layers.Dense(2) (X)
            model = Model(inputs = inp, outputs = X, name = model_name)

        model.compile(loss = loss, optimizer = opt, metrics = metrics)

        # Print verbose output to console
        if verbose:
            self.__verbose_print(model, model_name, input_shape, opt, loss, metrics)

        return model

    #/////////////////////////////////////////////////////
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    ### Adapter Pooler Tenney
    #/////////////////////////////////////////////////////
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    def get_adapter_pooler_tenney(self, input_shape = (1, 1024, 26), gpu_device = "/gpu:0", verbose = True):
        r"""Returns the TensorFlow 2.2 implementation of Adapter Pooler Tenny linear model.
            tensor contraction inspired by Tenney et al. 2019 : https://arxiv.org/abs/1905.05950
            adapter pooler layer inspired by Houlsby et al. 2019 : https://arxiv.org/abs/1902.00751v2

        Args:
            input_shape (tuple, optional): Shape of the input tensor. Defaults to (1, 1024, 26).
            gpu_device (str, optional): If GPU devices are available, defines which one to utilize. Defaults to "/gpu:0".
            verbose (bool, optional): Log details to console. Defaults to True.

        Returns:
           tf.keras.Model: returns a TensorFlow 2.20 model (compiled, untrained)
        """

        # Input validation
        self.__require_params(input_shape = input_shape)

        if (not gpu_device) or self.__GPU_count == 0:
            gpu_device = "/cpu:0"

        # Model hyperparameters and metadata
        model_name = 'Binary Classification Adapter Pooler Tenney'
        opt = Adam(lr = 1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
        loss = CategoricalCrossentropy(from_logits = True)
        metrics = ['accuracy']

        # Construct model & compile
        with tf.device(gpu_device):
            inp = layers.Input(input_shape, name = 'input_layer')
            inp_seq = inp[:,:,:,-1]
            X = BertConcat() (inp)
            X = tf.expand_dims(X, axis = -1, name ='expand_dims')
            X = AdapterPooler(386, shared_weights = True) (X)
            X = tf.reshape(X, (-1, X.shape[1], X.shape[2] * X.shape[3]))
            X = tf.concat([X, inp_seq], axis = 2)
            X = tf.squeeze(X, axis = 1)
            X = layers.Dense(2) (X)
            model = Model(inputs = inp, outputs = X, name = model_name)
        
        model.compile(loss = loss, optimizer = opt, metrics = metrics)

        # Print verbose output to console
        if verbose:
            self.__verbose_print(model, model_name, input_shape, opt, loss, metrics)

        return model

    #/////////////////////////////////////////////////////
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    ### Xception (Abbreviated)
    #/////////////////////////////////////////////////////
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    def get_xception_abbreviated(self, input_shape = (1, 1024, 26), gpu_device = "/gpu:0", verbose = True):
        r"""Returns the TensorFlow 2.2 implementation of Xception (Abbreviated).
            Inspired by Chollet 2017 : http://arxiv.org/abs/1610.02357

        Args:
            input_shape (tuple, optional): Shape of the input tensor. Defaults to (1, 1024, 26).
            gpu_device (str, optional): If GPU devices are available, defines which one to utilize. Defaults to "/gpu:0".
            verbose (bool, optional): Log details to console. Defaults to True.

        Returns:
           tf.keras.Model: returns a TensorFlow 2.20 model (compiled, untrained)
        """

        # Input validation
        self.__require_params(input_shape = input_shape)

        if (not gpu_device) or self.__GPU_count == 0:
            gpu_device = "/cpu:0"

        # Model hyperparameters and metadata
        model_name = 'Binary Classification Xception (Abbreviated)'
        opt = Adam(lr = 1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
        loss = CategoricalCrossentropy(from_logits = True)
        metrics = ['accuracy']

        # Construct model & compile
        with tf.device(gpu_device):

            # input image size
            input_img = layers.Input(shape = input_shape, dtype = tf.float32)

            # Block 1
            x = Conv2D(64, (1, 3), strides=(1, 3), use_bias=False) (input_img)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(128, (1, 3), use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            residual = Conv2D(512, (1, 1), strides=(1, 2), padding='same', use_bias=False)(x)
            residual = BatchNormalization()(residual)

            # Block 2
            x = SeparableConv2D(256, (1, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(512, (1, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)

            # Block 2 Pool
            x = AveragePooling2D((1, 3), strides=(1, 2), padding='same')(x)
            x = layers.add([x, residual])

            # Fully Connected Layer
            x = GlobalAveragePooling2D()(x)

            x = layers.Dense(2, dtype = tf.float32, name = 'dense_2_final') (x)

            model = models.Model(input_img, x, name = model_name)

        model.compile(loss = loss, optimizer = opt, metrics = metrics)

        # Print verbose output to console
        if verbose:
            self.__verbose_print(model, model_name, input_shape, opt, loss, metrics)

        return model

    #/////////////////////////////////////////////////////
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    ### Xception
    #/////////////////////////////////////////////////////
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    def get_xception(self, input_shape = (1, 1024, 26), gpu_device = "/gpu:0", verbose = True):
        r"""Returns the TensorFlow 2.2 implementation of Xception for SQuAD v2 Binary Classification.
            Inspired by Chollet 2017 : http://arxiv.org/abs/1610.02357

        Args:
            input_shape (tuple, optional): Shape of the input tensor. Defaults to (1, 1024, 26).
            gpu_device (str, optional): If GPU devices are available, defines which one to utilize. Defaults to "/gpu:0".
            verbose (bool, optional): Log details to console. Defaults to True.

        Returns:
           tf.keras.Model: returns a TensorFlow 2.20 model (compiled, untrained)
        """

        # Input validation
        self.__require_params(input_shape = input_shape)

        if (not gpu_device) or self.__GPU_count == 0:
            gpu_device = "/cpu:0"

        # Model hyperparameters and metadata
        model_name = 'Binary Classification Xception'
        opt = Adam(lr = 1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
        loss = CategoricalCrossentropy(from_logits = True)
        metrics = ['accuracy']

        # Construct model & compile
        with tf.device(gpu_device):

            # input image size
            input_img = layers.Input(shape = input_shape, dtype = tf.float32)

            # Block 1
            x = Conv2D(32, (1, 3), strides=(1, 3), use_bias=False) (input_img)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(64, (1, 3), use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            residual = Conv2D(128, (1, 1), strides=(1, 2), padding='same', use_bias=False)(x)
            residual = BatchNormalization()(residual)

            # Block 2
            x = SeparableConv2D(128, (1, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(128, (1, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)

            # Block 2 Pool
            x = AveragePooling2D((1, 3), strides=(1, 2), padding='same')(x)
            x = layers.add([x, residual])

            residual = Conv2D(256, (1, 1), strides=(1, 2), padding='same', use_bias=False)(x)
            residual = BatchNormalization()(residual)

            # Block 3
            x = Activation('relu')(x)
            x = SeparableConv2D(256, (1, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(256, (1, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)

            # Block 3 Pool
            x = AveragePooling2D((1, 3), strides=(1, 2), padding='same')(x)
            x = layers.add([x, residual])

            residual = Conv2D(728, (1, 1), strides=(1, 2), padding='same', use_bias=False)(x)
            residual = BatchNormalization()(residual)

            # Block 4
            x = Activation('relu')(x)
            x = SeparableConv2D(728, (1, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(728, (1, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)

            x = AveragePooling2D((1, 3), strides=(1, 2), padding='same')(x)
            x = layers.add([x, residual])

            # Block 5 - 12
            for i in range(8):
                residual = x

                x = Activation('relu')(x)
                x = SeparableConv2D(728, (1, 3), padding='same', use_bias=False)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = SeparableConv2D(728, (1, 3), padding='same', use_bias=False)(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = SeparableConv2D(728, (1, 3), padding='same', use_bias=False)(x)
                x = BatchNormalization()(x)

                x = layers.add([x, residual])

            residual = Conv2D(1024, (1, 1), strides=(1, 2), padding='same', use_bias=False)(x)
            residual = BatchNormalization()(residual)

            # Block 13
            x = Activation('relu')(x)
            x = SeparableConv2D(728, (1, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(1024, (1, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)

            # Block 13 Pool
            x = AveragePooling2D((1, 3), strides=(1, 2), padding='same')(x)
            x = layers.add([x, residual])

            # Block 14
            x = SeparableConv2D(1536, (1, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # Block 14 part 2
            x = SeparableConv2D(2048, (1, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
                    
            # Fully Connected Layer
            x = GlobalAveragePooling2D()(x)

            x = layers.Dense(2, dtype = tf.float32, name = 'dense_2_final') (x)

            model = models.Model(input_img, x, name = model_name)

        model.compile(loss = loss, optimizer = opt, metrics = metrics)

        # Print verbose output to console
        if verbose:
            self.__verbose_print(model, model_name, input_shape, opt, loss, metrics)

        return model

    #/////////////////////////////////////////////////////
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    ### Xception (Abbreviated w/ CLS Residual)
    #/////////////////////////////////////////////////////
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    def get_xception_abbreviated_clsresidual(self, input_shape = (1, 1024, 26), gpu_device = "/gpu:1"):
        r"""Returns the TensorFlow 2.2 implementation of Xception (Abbreviated w/ CLS Residual).
            Inspired by Chollet 2017 : http://arxiv.org/abs/1610.02357

        Args:
            input_shape (tuple, optional): Shape of the input tensor. Defaults to (1, 1024, 26).
            gpu_device (str, optional): If GPU devices are available, defines which one to utilize. Defaults to "/gpu:0".
            verbose (bool, optional): Log details to console. Defaults to True.

        Returns:
           tf.keras.Model: returns a TensorFlow 2.20 model (compiled, untrained)
        """

        # Input validation
        self.__require_params(input_shape = input_shape)

        if (not gpu_device) or self.__GPU_count == 0:
            gpu_device = "/cpu:0"

        # Model hyperparameters and metadata
        model_name = 'Binary Classification Xception (Abbreviated w/ CLS Residual)'
        opt = Adam(lr = 1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
        loss = CategoricalCrossentropy(from_logits = True)
        metrics = ['accuracy']

        # Construct model & compile
        with tf.device(gpu_device):

            # input image size
            input_img = layers.Input(shape = input_shape, dtype = tf.float32)

            # pull the last channel layer for residual connection layer
            inp_seq = input_img[:,:,:,-1]
            inp_seq = tf.squeeze(inp_seq, axis = 1)

            # Block 1
            x = Conv2D(64, (1, 3), strides=(1, 3), use_bias=False) (input_img)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(128, (1, 3), use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

            residual = Conv2D(512, (1, 1), strides=(1, 2), padding='same', use_bias=False)(x)
            residual = BatchNormalization()(residual)

            # Block 2
            x = SeparableConv2D(256, (1, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(512, (1, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)

            # Block 2 Pool
            x = AveragePooling2D((1, 3), strides=(1, 2), padding='same')(x)
            x = layers.add([x, residual])

            # Fully Connected Layer
            x = GlobalAveragePooling2D()(x)
            
            # add the skip level residual back to the last CLS token
            x = layers.concatenate([x, inp_seq])

            x = layers.Dense(2, dtype = tf.float32, name = 'dense_2_final') (x)

            model = models.Model(input_img, x, name = model_name)

        model.compile(loss = loss, optimizer = opt, metrics = metrics)

        # Print verbose output to console
        if verbose:
            self.__verbose_print(model, model_name, input_shape, opt, loss, metrics)

        return model

    #/////////////////////////////////////////////////////
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    ### Xception (Abbreviated w/ CLS Residual)
    #/////////////////////////////////////////////////////
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    def get_adapaterpooling_meanavg(self, input_shape = (1, 1024, 26), gpu_device = "/gpu:0", verbose = True):
        r"""Returns the TensorFlow 2.2 implementation of Adapter Pooler Mean Average model.
            tensor contraction along the channel dimension perfomed using simple mean averaging.
            adapter pooler layer inspired by Houlsby et al. 2019 : https://arxiv.org/abs/1902.00751v2

        Args:
            input_shape (tuple, optional): Shape of the input tensor. Defaults to (1, 1024, 26).
            gpu_device (str, optional): If GPU devices are available, defines which one to utilize. Defaults to "/gpu:0".
            verbose (bool, optional): Log details to console. Defaults to True.

        Returns:
           tf.keras.Model: returns a TensorFlow 2.20 model (compiled, untrained)
        """

        # Input validation
        self.__require_params(input_shape = input_shape)

        if (not gpu_device) or self.__GPU_count == 0:
            gpu_device = "/cpu:0"

        # Model hyperparameters and metadata
        model_name = 'Binary Classification Adapter Pooler Mean Average'
        opt = Adam(lr = 1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
        loss = CategoricalCrossentropy(from_logits = True)
        metrics = ['accuracy']

        # Construct model & compile        
        with tf.device(gpu_device):

            inp = layers.Input(input_shape, name = 'input_layer')
            inp_seq = inp[:,:,:,-1]
            X = MeanConcat() (inp)
            X = tf.expand_dims(X, axis = -1, name ='expand_dims')
            X = AdapterPooler(386, shared_weights = True)(X)
            X = tf.reshape(X, (-1, X.shape[1], X.shape[2] * X.shape[3]))
            X = tf.concat([X, inp_seq], axis = 2)
            X = tf.squeeze(X, axis = 1)
            X = layers.Dense(2)(X)

            model = Model(inputs = inp, outputs = X, name = model_name)

        model.compile(loss = loss, optimizer = opt, metrics = metrics)

        # Print verbose output to console
        if verbose:
            self.__verbose_print(model, model_name, input_shape, opt, loss, metrics)

        return model


###########################################################################################################
## CLASS CONTAINING SPAN ANNOTATION MODELS
###########################################################################################################
class QnAModels(object):

    def __init__(self, **kwargs):
        self.__GPU_count = len(tf.config.list_physical_devices('GPU'))

    ######################################################
    ### Private Methods
    ######################################################

    # validate required input parameter values aren't set to None
    @staticmethod
    def __require_params(**kwargs):
        
        needed_args = [key for key,value in kwargs.items() if value is None]
        if len(needed_args) > 0:
            raise ValueError("If running in training, must specify following outputs: %s" %(', '.join(needed_args)))
    
        return

    def __verbose_print(self, model, model_name, input_shape, opt, loss, metrics):

        print("".join(["\n", "*" * 100, "\nModel Details\n", "*" * 100, "\n"]))
        print(f"Model Name: {model_name}")
        print(f"Optimizer Details:  name = {opt.get_config()['name']},  learning rate = {opt.get_config()['learning_rate']}")
        print(f"Loss Details:  name = {loss.get_config()['name']}, from_logits = {loss.get_config()['from_logits']}")
        print(f"Input Shape: {tuple(input_shape)}")
        print(f"Metrics: {"".join(metrics)}")
        print("*" * 100)
        print(model.summary())
        print("*" * 100, "\n")

        return

    ######################################################
    ### Public Methods
    ######################################################

    #/////////////////////////////////////////////////////
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    ### Sample Model
    #/////////////////////////////////////////////////////
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    def get_sample_model(self, input_shape = (386, 1024, 26), gpu_device = "/gpu:0", verbose = True):
        r"""Returns the sample model for QnA Span Annotation task.
            adapter pooler layer inspired by Houlsby et al. 2019 : https://arxiv.org/abs/1902.00751v2

        Args:
            input_shape (tuple, optional): Shape of the input tensor. Defaults to (1, 1024, 26).
            gpu_device (str, optional): If GPU devices are available, defines which one to utilize. Defaults to "/gpu:0".
            verbose (bool, optional): Log details to console. Defaults to True.

        Returns:
           tf.keras.Model: returns a TensorFlow 2.20 model (compiled, untrained)
        """

        # Input validation
        self.__require_params(input_shape = input_shape)

        if (not gpu_device) or self.__GPU_count == 0:
            gpu_device = "/cpu:0"

        # Model hyperparameters and metadata
        model_name = 'QnA Sample Model'
        opt = Adam(lr = 1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
        loss = CategoricalCrossentropy(from_logits = True)
        metrics = ['accuracy']

        # Construct model & compile        
        with tf.device(gpu_device):

            inp = layers.Input(input_shape, name = 'input_layer')
            inp_seq = inp[:,:,:,-1]
            X = MeanConcat() (inp)
            X = tf.expand_dims(X, axis = -1, name ='expand_dims')
            X = AdapterPooler(386, shared_weights = True)(X)
            X = tf.reshape(X, (-1, X.shape[1], X.shape[2] * X.shape[3]))
            X = tf.concat([X, inp_seq], axis = 2)
            X = tf.squeeze(X, axis = 1)
            X = layers.Dense(2)(X)

            model = Model(inputs = inp, outputs = X, name = model_name)

        model.compile(loss = loss, optimizer = opt, metrics = metrics)

        # Print verbose output to console
        if verbose:
            self.__verbose_print(model, model_name, input_shape, opt, loss, metrics)

        return model