###########################################################################################################
## IMPORTS
###########################################################################################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Activation, Convolution2D, Conv2D, LocallyConnected2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dense, Dropout, Input, concatenate, add, Add, ZeroPadding2D, GlobalMaxPooling2D, DepthwiseConv2D, LeakyReLU, ELU, ReLU
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

#from keras.activations import linear, elu, tanh, relu
from tensorflow.keras import metrics, losses, initializers, backend
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.initializers import glorot_uniform, Constant, lecun_uniform
from tensorflow.keras import backend as K

#os.environ["PATH"] += os.pathsep + "C:/ProgramData/Anaconda3/GraphViz/bin/"
#os.environ["PATH"] += os.pathsep + "C:/Anaconda/Graphviz2.38/bin/"

from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split

import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

tf.get_logger().setLevel('ERROR')

physical_devices = tf.config.list_physical_devices('GPU')
#for pd_dev in range(len(physical_devices)):
#    tf.config.experimental.set_memory_growth(physical_devices[pd_dev], True)

###########################################################################################################
## PLOTTING PALETTE
###########################################################################################################

# Create a dict object containing U.C. Berkeley official school colors for plot palette
# reference : https://alumni.berkeley.edu/brand/color-palette

berkeley_palette = {'berkeley_blue'     : '#003262',
                    'california_gold'   : '#FDB515',
                    'metallic_gold'     : '#BC9B6A',
                    'founders_rock'     : '#2D637F',
                    'medalist'          : '#E09E19',
                    'bay_fog'           : '#C2B9A7',
                    'lawrence'          : '#00B0DA',
                    'sather_gate'       : '#B9D3B6',
                    'pacific'           : '#53626F',
                    'soybean'           : '#9DAD33',
                    'california_purple' : '#5C3160',
                    'south_hall'        : '#6C3302'}

###########################################################################################################
## CLASS CONTAINING MODEL ZOO
###########################################################################################################
class Models(object):

    def __init__(self, model_path, **kwargs):
        super(Models, self).__init__(** kwargs)

        # validate that the constructor parameters were provided by caller
        if (not model_path):
            raise RuntimeError('path to model files must be provided on initialization.')

        # ensure all are string snd leading/trailing whitespace removed
        model_path = str(model_path).replace('\\', '/').strip()
        if (not model_path.endswith('/')): model_path = ''.join((model_path, '/'))

        # validate the existence of the data path
        if (not os.path.isdir(model_path)):
            raise RuntimeError(f"Models path specified '{model_path}' is invalid.")

        self.__models_path = model_path
        self.__GPU_count = len(tf.config.list_physical_devices('GPU'))
        self.__MIN_early_stopping = 10

    #------------------------------------------------
    # Private Methods
    #------------------------------------------------

    # plotting method for keras history arrays
    def __plot_keras_history(self, history, metric, model_name, file_name, verbose = False):
            # Plot the performance of the model training
            fig = plt.figure(figsize=(15,8),dpi=80)
            ax = fig.add_subplot(121)

            ax.plot(history.history[metric][1:], color = berkeley_palette['founders_rock'], label = 'Train',
                marker = 'o', markersize = 4, alpha = 0.9)
            ax.plot(history.history["".join(["val_",metric])][1:], color = berkeley_palette['medalist'], label = 'Validation',
                marker = 'o', markersize = 4, alpha = 0.9)
            ax.set_title(" ".join(['Model Performance',"(" + model_name + ")"]) + "\n",
                color = berkeley_palette['berkeley_blue'], fontsize = 15, fontweight = 'bold')
            ax.spines["top"].set_alpha(.0)
            ax.spines["bottom"].set_alpha(.3)
            ax.spines["right"].set_alpha(.0)
            ax.spines["left"].set_alpha(.3)
            ax.set_xlabel("Epoch", fontsize = 12, horizontalalignment='right', x = 1.0, color = berkeley_palette['berkeley_blue'])
            ax.set_ylabel(metric, fontsize = 12, horizontalalignment='right', y = 1.0, color = berkeley_palette['berkeley_blue'])
            plt.legend(loc = 'upper right')

            ax = fig.add_subplot(122)

            ax.plot(history.history['loss'][1:], color = berkeley_palette['founders_rock'], label = 'Train',
                marker = 'o', markersize = 4, alpha = 0.9)
            ax.plot(history.history["".join(["val_loss"])][1:], color = berkeley_palette['medalist'], label = 'Validation',
                marker = 'o', markersize = 4, alpha = 0.9)
            ax.set_title(" ".join(['Model Performance',"(" + model_name + ")"]) + "\n",
                color = berkeley_palette['berkeley_blue'], fontsize = 15, fontweight = 'bold')
            ax.spines["top"].set_alpha(.0)
            ax.spines["bottom"].set_alpha(.3)
            ax.spines["right"].set_alpha(.0)
            ax.spines["left"].set_alpha(.3)
            ax.set_xlabel("Epoch", fontsize = 12, horizontalalignment='right', x = 1.0, color = berkeley_palette['berkeley_blue'])
            ax.set_ylabel("Loss", fontsize = 12, horizontalalignment='right', y = 1.0, color = berkeley_palette['berkeley_blue'])
            plt.legend(loc = 'upper right')

            plt.tight_layout()
            plt.savefig(file_name, dpi=300)
            if verbose: print("Training plot file saved to '%s'." % file_name)
            plt.close()

    # load Keras model files from json / h5
    def __load_keras_model(self, model_name, model_file, model_json, verbose = False):
        """Loads a Keras model from disk"""

        if not os.path.isfile(model_file):
            raise RuntimeError(f"Model file '{model_file}' does not exist; exiting inferencing.")

        if not os.path.isfile(model_json):
            raise RuntimeError(f"Model file '{model_json}' does not exist; exiting inferencing.")

        # load model file
        if verbose: print(f"Retrieving model: {model_name}...")
        json_file = open(model_json, "r")
        model_json_data = json_file.read()
        json_file.close()
        model = model_from_json(model_json_data)
        model.load_weights(model_file)

        return model

    # resnet identity block builder
    def __identity_block(self, model, kernel_size, filters, stage, block):
        """modularized identity block for resnet"""
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1),
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2a')(model)
        x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters2, kernel_size,
                        padding='same',
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1),
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

        x = add([x, model])
        x = Activation('relu')(x)
        return x

    # resnet conv block builder
    def __conv_block(self, model, kernel_size, filters, stage, block, strides=(2, 2)):
        """conv block builder for resnet"""
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, (1, 1), strides=strides,
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2a')(model)
        x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
        x =Activation('relu')(x)

        x = Conv2D(filters2, kernel_size, padding='same',
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(filters3, (1, 1),
                        kernel_initializer='he_normal',
                        name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

        shortcut = Conv2D(filters3, (1, 1), strides=strides,
                                kernel_initializer='he_normal',
                                name=conv_name_base + '1')(model)
        shortcut = BatchNormalization(
            axis=3, name=bn_name_base + '1')(shortcut)

        x = add([x, shortcut])
        x = Activation('relu')(x)

        return x

    # create a layerable inception module
    def __inception_module(self, model, filters_1x1, filters_3x3_reduce, filters_3x3,
            filters_5x5_reduce, filters_5x5, filters_pool_proj, kernel_init, bias_init, name = None):
        """modularized inception block for layering"""

        # Connection Layer 1 (1x1)
        conv_1x1 = Convolution2D(filters_1x1, (1, 1), padding = 'same', activation = 'relu',
            kernel_initializer = kernel_init, bias_initializer = bias_init) (model)

        # Connection Layer 2 (3x3)
        conv_3x3 = Convolution2D(filters_3x3_reduce, (1, 1), padding = 'same', activation = 'relu',
            kernel_initializer = kernel_init, bias_initializer = bias_init) (model)
        conv_3x3 = Convolution2D(filters_3x3, (3, 3), padding = 'same', activation = 'relu',
            kernel_initializer = kernel_init, bias_initializer = bias_init) (conv_3x3)

        # Connection Layer 3 (5x5)
        conv_5x5 = Convolution2D(filters_5x5_reduce, (1, 1), padding = 'same', activation = 'relu',
            kernel_initializer = kernel_init, bias_initializer = bias_init) (model)
        conv_5x5 = Convolution2D(filters_5x5, (5, 5), padding = 'same', activation = 'relu',
            kernel_initializer = kernel_init, bias_initializer = bias_init) (conv_5x5)

        # Connection Layer 4 (pool)
        pool_proj = MaxPooling2D((3, 3), strides = (1, 1), padding = 'same') (model)
        pool_proj = Convolution2D(filters_pool_proj, (1, 1), padding = 'same', activation = 'relu',
            kernel_initializer = kernel_init, bias_initializer = bias_init) (pool_proj)

        # Concatenation layer
        output = concatenate(inputs = [conv_1x1, conv_3x3, conv_5x5, pool_proj], axis = 3, name = name)

        return output

    # return an InceptionV3 output tensor after applying Conv2D and BatchNormalization
    def __conv2d_bn(self, x, filters, num_row, num_col, padding = 'same', strides = (1, 1), name = None):

        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        bn_axis = 3

        x = Convolution2D(filters, (num_row, num_col), strides = strides,
            padding = padding, use_bias = False, name = conv_name) (x)

        x = BatchNormalization(axis = bn_axis, scale = False, name = bn_name) (x)
        x = ReLU(name = name) (x)

        return x

    # a residual block for resnext
    def __resnext_block(self, x, filters, kernel_size = 3, stride = 1, groups = 32, conv_shortcut = True, name = None):

        if conv_shortcut is True:
            shortcut = Conv2D((64 // groups) * filters, 1, strides = stride, use_bias = False, name = name + '_0_conv') (x)
            shortcut = BatchNormalization(axis = 3, epsilon=1.001e-5, name = name + '_0_bn') (shortcut)
        else:
            shortcut = x

        x = Conv2D(filters, 1, use_bias = False, name = name + '_1_conv') (x)
        x = BatchNormalization(axis = 3, epsilon = 1.001e-5, name = name + '_1_bn') (x)
        x = Activation('relu', name = name + '_1_relu') (x)

        c = filters // groups
        x = ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
        x = DepthwiseConv2D(kernel_size, strides = stride, depth_multiplier = c, use_bias = False, name = name + '_2_conv') (x)
        kernel = np.zeros((1, 1, filters * c, filters), dtype = np.float32)
        for i in range(filters):
            start = (i // c) * c * c + i % c
            end = start + c * c
            kernel[:, :, start:end:c, i] = 1.
        x = Conv2D(filters, 1, use_bias = False, trainable = False, kernel_initializer = {'class_name': 'Constant','config': {'value': kernel}}, name = name + '_2_gconv') (x)
        x = BatchNormalization(axis=3, epsilon = 1.001e-5, name = name + '_2_bn') (x)
        x = Activation('relu', name=name + '_2_relu') (x)

        x = Conv2D((64 // groups) * filters, 1, use_bias = False, name = name + '_3_conv') (x)
        x = BatchNormalization(axis = 3, epsilon=1.001e-5, name = name + '_3_bn') (x)

        x = Add(name = name + '_add') ([shortcut, x])
        x = Activation('relu', name = name + '_out') (x)
        return x

    # a set of stacked residual blocks for ResNeXt
    def __resnext_stack(self, x, filters, blocks, stride1 = 2, groups = 32, name = None, dropout = None):
        x = self.__resnext_block(x, filters, stride = stride1, groups = groups, name = name + '_block1')
        for i in range(2, blocks + 1):
            x = self.__resnext_block(x, filters, groups = groups, conv_shortcut = False,
                    name = name + '_block' + str(i))
        if not dropout is None:
            x = Dropout(dropout) (x)
        return x

    def __bn_relu(self, x, bn_name = None, relu_name = None):
        norm = BatchNormalization(axis = 3, name = bn_name) (x)
        return Activation("relu", name = relu_name) (norm)

    def __bn_relu_conv(self, **conv_params):

        filters = conv_params["filters"]
        kernel_size = conv_params["kernel_size"]
        strides = conv_params.setdefault("strides", (1, 1))
        dilation_rate = conv_params.setdefault("dilation_rate", (1, 1))
        conv_name = conv_params.setdefault("conv_name", None)
        bn_name = conv_params.setdefault("bn_name", None)
        relu_name = conv_params.setdefault("relu_name", None)
        kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
        padding = conv_params.setdefault("padding", "same")
        kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

        def f(x):
            activation = self.__bn_relu(x, bn_name = bn_name, relu_name = relu_name)
            return Conv2D(filters = filters, kernel_size = kernel_size,
                        strides = strides, padding = padding,
                        dilation_rate = dilation_rate,
                        kernel_initializer = kernel_initializer,
                        kernel_regularizer = kernel_regularizer,
                        name = conv_name) (activation)

        return f

    def __conv_bn_relu(self, **conv_params):

        filters = conv_params["filters"]
        kernel_size = conv_params["kernel_size"]
        strides = conv_params.setdefault("strides", (1, 1))
        dilation_rate = conv_params.setdefault("dilation_rate", (1, 1))
        conv_name = conv_params.setdefault("conv_name", None)
        bn_name = conv_params.setdefault("bn_name", None)
        relu_name = conv_params.setdefault("relu_name", None)
        kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
        padding = conv_params.setdefault("padding", "same")
        kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

        def f(x):
            x = Conv2D(filters = filters, kernel_size = kernel_size,
                    strides = strides, padding = padding,
                    dilation_rate = dilation_rate,
                    kernel_initializer = kernel_initializer,
                    kernel_regularizer = kernel_regularizer,
                    name = conv_name) (x)
            return self.__bn_relu(x, bn_name = bn_name, relu_name = relu_name)

        return f

    def __block_name_base(self, stage, block):
        if block < 27:
            block = '%c' % (block + 97)  # 97 is the ascii number for lowercase 'a'
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        return conv_name_base, bn_name_base

    def __shortcut(self, input_feature, residual, conv_name_base = None, bn_name_base = None):
        input_shape = K.int_shape(input_feature)
        residual_shape = K.int_shape(residual)
        stride_width = int(round(input_shape[1] / residual_shape[1]))
        stride_height = int(round(input_shape[2] / residual_shape[2]))
        equal_channels = input_shape[3] == residual_shape[3]

        shortcut = input_feature
        # 1 X 1 conv if shape is different. Else identity.
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            print('reshaping via a convolution...')
            if conv_name_base is not None:
                conv_name_base = conv_name_base + '1'
            shortcut = Conv2D(filters=residual_shape[3],
                            kernel_size=(1, 1),
                            strides=(stride_width, stride_height),
                            padding="valid",
                            kernel_initializer="he_normal",
                            kernel_regularizer=l2(0.0001),
                            name=conv_name_base)(input_feature)
            if bn_name_base is not None:
                bn_name_base = bn_name_base + '1'
            shortcut = BatchNormalization(axis=3,
                                        name=bn_name_base)(shortcut)

        return add([shortcut, residual])

    def __basic_block(self, filters, stage, block, transition_strides = (1, 1),
                    dilation_rate = (1, 1), is_first_block_of_first_layer = False, dropout = None,
                    residual_unit = None):

        def f(input_features):
            conv_name_base, bn_name_base = self.__block_name_base(stage, block)
            if is_first_block_of_first_layer:
                # don't repeat bn->relu since we just did bn->relu->maxpool
                x = Conv2D(filters = filters, kernel_size = (3, 3),
                        strides = transition_strides, dilation_rate = dilation_rate,
                        padding = "same", kernel_initializer = "he_normal", kernel_regularizer = l2(1e-4),
                        name = conv_name_base + '2a') (input_features)
            else:
                x = residual_unit(filters = filters, kernel_size = (3, 3),
                                strides = transition_strides,
                                dilation_rate = dilation_rate,
                                conv_name_base = conv_name_base + '2a',
                                bn_name_base = bn_name_base + '2a') (input_features)

            if dropout is not None:
                x = Dropout(dropout) (x)

            x = residual_unit(filters = filters, kernel_size = (3, 3),
                            conv_name_base = conv_name_base + '2b',
                            bn_name_base = bn_name_base + '2b') (x)

            return self.__shortcut(input_features, x)

        return f

    def __bottleneck(self, filters, stage, block, transition_strides = (1, 1),
                dilation_rate = (1, 1), is_first_block_of_first_layer = False, dropout = None,
                residual_unit = None):
        """Bottleneck architecture for > 34 layer resnet.
        Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
        Returns:
            A final conv layer of filters * 4
        """
        def f(input_feature):
            conv_name_base, bn_name_base = self.__block_name_base(stage, block)
            if is_first_block_of_first_layer:
                # don't repeat bn->relu since we just did bn->relu->maxpool
                x = Conv2D(filters=filters, kernel_size=(1, 1),
                        strides=transition_strides,
                        dilation_rate=dilation_rate,
                        padding="same",
                        kernel_initializer="he_normal",
                        kernel_regularizer=l2(1e-4),
                        name=conv_name_base + '2a')(input_feature)
            else:
                x = residual_unit(filters=filters, kernel_size=(1, 1),
                                strides=transition_strides,
                                dilation_rate=dilation_rate,
                                conv_name_base=conv_name_base + '2a',
                                bn_name_base=bn_name_base + '2a')(input_feature)

            if dropout is not None:
                x = Dropout(dropout)(x)

            x = residual_unit(filters=filters, kernel_size=(3, 3),
                            conv_name_base=conv_name_base + '2b',
                            bn_name_base=bn_name_base + '2b')(x)

            if dropout is not None:
                x = Dropout(dropout)(x)

            x = residual_unit(filters=filters * 4, kernel_size=(1, 1),
                            conv_name_base=conv_name_base + '2c',
                            bn_name_base=bn_name_base + '2c')(x)

            return self.__shortcut(input_feature, x)

        return f

    # builds a residual block for resnet with repeating bottleneck blocks
    def __residual_block(self, block_function, filters, blocks, stage, transition_strides = None, transition_dilation_rates = None,
        dilation_rates = None, is_first_layer = False, dropout = None, residual_unit = None):

        if transition_dilation_rates is None:
            transition_dilation_rates = [(1, 1)] * blocks
        if transition_strides is None:
            transition_strides = [(1, 1)] * blocks
        if dilation_rates is None:
            dilation_rates = [1] * blocks

        def f(x):
            for i in range(blocks):
                is_first_block = is_first_layer and i == 0
                x = block_function(filters=filters, stage=stage, block=i,
                                transition_strides=transition_strides[i],
                                dilation_rate=dilation_rates[i],
                                is_first_block_of_first_layer=is_first_block,
                                dropout=dropout,
                                residual_unit=residual_unit)(x)
            return x

        return f

    ######################################################
    ######################################################
    ######################################################
    ### MODEL ZOO
    ######################################################
    ######################################################
    ######################################################

    #-------------------------------------------------------------
    # Inception V1
    # Inspired by : https://arxiv.org/abs/1409.4842
    #-------------------------------------------------------------

    @staticmethod
    def __require(**kwargs):
        needed_args = [key for key,value in kwargs.items() if value is None]
        raise ValueError("If running in training, must specify following outputs: %s" %(', '.join(needed_args)))

    def get_keras_inception_v1(self,
                               input_shape,
                               return_model_only = True,
                               X = None,
                               Y = None,
                               batch_size = None,
                               epoch_count = None,
                               val_split = 0.1,
                               shuffle = True,
                               recalculate_pickle = True,
                               X_val = None,
                               Y_val = None,
                               verbose = False,
                               include_head = False):

        if not return_model_only:
            self.__require(X=X, Y=Y, batch_size=batch_size, epoch_count=epoch_count)

        __MODEL_NAME = "Keras_Inception_v1"
        __MODEL_FNAME_PREFIX = "KERAS_INCEPTION/"

        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            os.makedirs(nested_dir)

        __model_file_MAIN_name = "".join([nested_dir, __MODEL_NAME, "_MAIN.h5"])
        __model_file_AUX1_name = "".join([nested_dir, __MODEL_NAME, "_AUX1.h5"])
        __model_file_AUX2_name = "".join([nested_dir, __MODEL_NAME, "_AUX2.h5"])

        __model_json_file = "".join([nested_dir, __MODEL_NAME, ".json"])
        __model_architecture_plot_file = "".join([nested_dir, __MODEL_NAME, "_plot.png"])
        __history_params_file = "".join([nested_dir, __MODEL_NAME, "_params.csv"])
        __history_performance_file = "".join([nested_dir, __MODEL_NAME, "_history.csv"])
        __history_plot_file_main = "".join([nested_dir, __MODEL_NAME, "_main_output_plot.png"])
        __history_plot_file_auxilliary1 = "".join([nested_dir, __MODEL_NAME, "_auxilliary_output_1_plot.png"])
        __history_plot_file_auxilliary2 = "".join([nested_dir, __MODEL_NAME, "_auxilliary_output_2_plot.png"])

        if verbose: print(f"Retrieving model: {__MODEL_NAME}...")

        # Create or load the model
        if (not os.path.isfile(__model_file_MAIN_name)) or (not os.path.isfile(__model_file_AUX1_name)) or (not os.path.isfile(__model_file_AUX2_name)) or (not os.path.isfile(__model_json_file)) or recalculate_pickle:
            if verbose: print(f"Pickle file for '{__MODEL_NAME}' MODEL not found or skipped by caller.")

            kernel_init = glorot_uniform()
            bias_init = Constant(value = 0.2)

            if self.__GPU_count > 1: dev = "/cpu:0"
            else: dev = "/gpu:0"
            with tf.device(dev):

                # Input image shape (H, W, C)
                input_img = Input(shape=input_shape)

                # Top Layer (Begin MODEL)
                model = Convolution2D(filters = 64, kernel_size = (7, 7), padding = 'same', strides = (2, 2),
                    activation = 'relu', name = 'conv_1_7x7/2', kernel_initializer = kernel_init,
                    bias_initializer = bias_init) (input_img)
                model = MaxPooling2D((3, 3), padding = 'same', strides = (2, 2), name = 'max_pool_1_3x3/2') (model)
                model = Convolution2D(64, (1, 1), padding = 'same', strides = (1, 1), activation = 'relu', name = 'conv_2a_3x3/1') (model)
                model = Convolution2D(192, (3, 3), padding = 'same', strides = (1, 1), activation = 'relu', name = 'conv_2b_3x3/1') (model)
                model = MaxPooling2D((3, 3), padding = 'same', strides = (2, 2), name = 'max_pool_2_3x3/2') (model)

                # Inception Module
                model = self.__inception_module(model,
                    filters_1x1 = 64,
                    filters_3x3_reduce = 96,
                    filters_3x3 = 128,
                    filters_5x5_reduce = 16,
                    filters_5x5 = 32,
                    filters_pool_proj = 32,
                    kernel_init = kernel_init,
                    bias_init = bias_init,
                    name = 'inception_3a')

                # Inception Module
                model = self.__inception_module(model,
                    filters_1x1 = 128,
                    filters_3x3_reduce = 128,
                    filters_3x3 = 192,
                    filters_5x5_reduce = 32,
                    filters_5x5 = 96,
                    filters_pool_proj = 64,
                    kernel_init = kernel_init,
                    bias_init = bias_init,
                    name = 'inception_3b')

                model = MaxPooling2D((3, 3), padding = 'same', strides = (2, 2), name= 'max_pool_3_3x3/2') (model)

                # Inception Module
                model = self.__inception_module(model,
                    filters_1x1 = 192,
                    filters_3x3_reduce = 96,
                    filters_3x3 = 208,
                    filters_5x5_reduce = 16,
                    filters_5x5 = 48,
                    filters_pool_proj = 64,
                    kernel_init = kernel_init,
                    bias_init = bias_init,
                    name = 'inception_4a')

                # CDB 3/5 DROPOUT ADDED
                model = Dropout(0.2) (model)

                # Begin MODEL1 (auxillary output)
                #Our shape is already too small in the encoder dimension
                #model1 = AveragePooling2D((5, 5), padding = 'same', strides = 3, name= 'avg_pool_4_5x5/2') (model)

                model1 = Convolution2D(128, (1, 1), padding = 'same', activation = 'relu') (model)

                if include_head:
                    model1 = Flatten() (model1)
                    model1 = Dense(1024, activation = 'relu') (model1)
                    model1 = Dropout(0.3) (model1)
                    model1 = Dense(30, name = 'auxilliary_output_1') (model1)

                # Resume MODEL w/ Inception
                model = self.__inception_module(model,
                    filters_1x1 = 160,
                    filters_3x3_reduce = 112,
                    filters_3x3 = 224,
                    filters_5x5_reduce = 24,
                    filters_5x5 = 64,
                    filters_pool_proj = 64,
                    kernel_init = kernel_init,
                    bias_init = bias_init,
                    name='inception_4b')

                model = self.__inception_module(model,
                    filters_1x1 = 128,
                    filters_3x3_reduce = 128,
                    filters_3x3 = 256,
                    filters_5x5_reduce = 24,
                    filters_5x5 = 64,
                    filters_pool_proj = 64,
                    kernel_init = kernel_init,
                    bias_init = bias_init,
                    name='inception_4c')

                model = self.__inception_module(model,
                    filters_1x1 = 112,
                    filters_3x3_reduce = 144,
                    filters_3x3 = 288,
                    filters_5x5_reduce = 32,
                    filters_5x5 = 64,
                    filters_pool_proj = 64,
                    kernel_init = kernel_init,
                    bias_init = bias_init,
                    name='inception_4d')

                # CDB : 3/5 DROPOUT ADDED
                model = Dropout(0.2) (model)

                # Begin MODEL2 (auxilliary output)

                #Our shape is already too small in the encoder dimension
                #model2 = AveragePooling2D((5, 5), strides = 3) (model)
                model2 = Convolution2D(128, (1, 1), padding = 'same', activation = 'relu') (model)

                if include_head:
                    model2 = Flatten() (model2)
                    model2 = Dense(1024, activation = 'relu') (model2)
                    model2 = Dropout(0.3) (model2)
                    model2 = Dense(30, name = 'auxilliary_output_2') (model2)

                # Resume MODEL w/ Inception
                model = self.__inception_module(model,
                    filters_1x1 = 256,
                    filters_3x3_reduce = 160,
                    filters_3x3 = 320,
                    filters_5x5_reduce = 32,
                    filters_5x5 = 128,
                    filters_pool_proj = 128,
                    kernel_init = kernel_init,
                    bias_init = bias_init,
                    name='inception_4e')

                model = MaxPooling2D((3, 3), padding = 'same', strides = (2, 2), name = 'max_pool_4_3x3/2') (model)

                model = self.__inception_module(model,
                    filters_1x1 = 256,
                    filters_3x3_reduce = 160,
                    filters_3x3 = 320,
                    filters_5x5_reduce = 32,
                    filters_5x5 = 128,
                    filters_pool_proj = 128,
                    kernel_init = kernel_init,
                    bias_init = bias_init,
                    name='inception_5a')

                model = self.__inception_module(model,
                    filters_1x1 = 384,
                    filters_3x3_reduce = 192,
                    filters_3x3 = 384,
                    filters_5x5_reduce = 48,
                    filters_5x5 = 128,
                    filters_pool_proj = 128,
                    kernel_init = kernel_init,
                    bias_init = bias_init,
                    name='inception_5b')

                if include_head:
                    # Output Layer (Main)
                    model = GlobalAveragePooling2D(name = 'avg_pool_5_3x3/1') (model)
                    model = Dropout(0.3) (model)
                    model = Dense(30, name = 'main_output') (model)

                model = Model(input_img, [model, model1, model2], name = 'Inception')

            if return_model_only:
                return model

            if verbose: print(model.summary())

            #Set up model training parameters
            act = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
            lss = 'mean_squared_error'
            mtrc = ['mae','mse']

            stop_at = np.max([int(0.1 * epoch_count), self.__MIN_early_stopping])
            es = EarlyStopping(patience = stop_at, verbose = verbose)

            cp_main = ModelCheckpoint(filepath = __model_file_MAIN_name, verbose = verbose, save_best_only = True,
                mode = 'min', monitor = 'val_main_output_mae')
            cp_aux1 = ModelCheckpoint(filepath = __model_file_AUX1_name, verbose = verbose, save_best_only = True,
                mode = 'min', monitor = 'val_auxilliary_output_1_mae')
            cp_aux2 = ModelCheckpoint(filepath = __model_file_AUX2_name, verbose = verbose, save_best_only = True,
                mode = 'min', monitor = 'val_auxilliary_output_2_mae')

            # Compile the model
            if self.__GPU_count > 1:
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    parallel_model = multi_gpu_model(model, gpus = self.__GPU_count)
                    parallel_model.compile(optimizer = act, loss = lss, metrics = mtrc)
            else:
                parallel_model = model
                parallel_model.compile(optimizer = act, loss = lss, metrics = mtrc)

            if (X_val is None) or (Y_val is None):
                history = parallel_model.fit(X, [Y, Y, Y], validation_split = val_split, batch_size = batch_size * self.__GPU_count,
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp_main, cp_aux1, cp_aux2], verbose = verbose)
            else:
                history = parallel_model.fit(X, [Y, Y, Y], validation_data = (X_val, [Y_val, Y_val, Y_val]), batch_size = batch_size * self.__GPU_count,
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp_main, cp_aux1, cp_aux2], verbose = verbose)

            # print and/or save a performance plot
            for m, f in zip(['main_output_mse', 'auxilliary_output_1_mse', 'auxilliary_output_2_mse'],
                [__history_plot_file_main, __history_plot_file_auxilliary1, __history_plot_file_auxilliary2]):
                try:
                    self.__plot_keras_history(history = history, metric = m, model_name = __MODEL_NAME,
                        file_name = f, verbose = False)
                except:
                    print("error during history plot generation; skipped.")
                    pass

            # save the model, parameters, and performance history
            model_json = parallel_model.to_json()
            with open(__model_json_file, "w") as json_file:
                json_file.write(model_json)
            hist_params = pd.DataFrame(history.params)
            hist_params.to_csv(__history_params_file)

            hist = pd.DataFrame(history.history)
            hist.to_csv(__history_performance_file)

            # save a plot of the model architecture
            try:
                plot_model(parallel_model, to_file = __model_architecture_plot_file, rankdir = 'TB',
                    show_shapes = True, show_layer_names = True, expand_nested = True, dpi = 300)
            except:
                print("error during model plot generation; skiopped.")
                pass

            if verbose: print("Model JSON, history, and parameters file saved.")

        else:
            if verbose: print(f"Loading history and params files for '{__MODEL_NAME}' model...")
            hist_params = pd.read_csv(__history_params_file)
            hist = pd.read_csv(__history_performance_file)

        if verbose: print(f"Loading pickle file for '{__MODEL_NAME}' model (MAIN) from file '{__model_file_MAIN_name}'")
        main_model = self.__load_keras_model(__MODEL_NAME, __model_file_MAIN_name, __model_json_file, verbose = verbose)

        if verbose: print(f"Loading pickle file for '{__MODEL_NAME}' model (AUX1) from file '{__model_file_AUX1_name}'")
        aux1_model = self.__load_keras_model(__MODEL_NAME, __model_file_AUX1_name, __model_json_file, verbose = verbose)

        if verbose: print(f"Loading pickle file for '{__MODEL_NAME}' model (AUX2) from file '{__model_file_AUX2_name}'")
        aux2_model = self.__load_keras_model(__MODEL_NAME, __model_file_AUX2_name, __model_json_file, verbose = verbose)

        return main_model, aux1_model, aux2_model, hist_params, hist

    # inferencing
    def predict_keras_inception_v1(self, X, verbose = False):

        __MODEL_NAME = "Keras_Inception_v1"
        __MODEL_FNAME_PREFIX = "KERAS_INCEPTION/"

        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            raise RuntimeError(f"Model path '{nested_dir}' does not exist; exiting inferencing.")

        __model_file_MAIN_name = "".join([nested_dir, __MODEL_NAME, "_MAIN.h5"])
        __model_file_AUX1_name = "".join([nested_dir, __MODEL_NAME, "_AUX1.h5"])
        __model_file_AUX2_name = "".join([nested_dir, __MODEL_NAME, "_AUX2.h5"])

        __model_json_file = "".join([nested_dir, __MODEL_NAME, ".json"])


        if (not os.path.isfile(__model_file_MAIN_name)) or (not os.path.isfile(__model_file_AUX1_name)) or (not os.path.isfile(__model_file_AUX2_name)) or (not os.path.isfile(__model_json_file)):
            raise RuntimeError("One or some of the following files are missing; prediction cancelled:\n\n'%s'\n'%s'\n'%s'\n'%s'\n\n" %
                (__model_file_MAIN_name, __model_file_AUX1_name, __model_file_AUX2_name, __model_json_file))

        # load the Keras model for the specified feature
        main_model = self.__load_keras_model(__MODEL_NAME, __model_file_MAIN_name, __model_json_file, verbose = verbose)
        aux1_model = self.__load_keras_model(__MODEL_NAME, __model_file_AUX1_name, __model_json_file, verbose = verbose)
        aux2_model = self.__load_keras_model(__MODEL_NAME, __model_file_AUX2_name, __model_json_file, verbose = verbose)

        # predict
        if verbose: print("Predicting %d (x,y) coordinates for 'MAIN' model file..." % len(X))
        Y_main = main_model.predict(X, verbose = verbose)
        Y_main_columns = [node.op.name for node in main_model.outputs]

        if verbose: print("Predicting %d (x,y) coordinates for 'AUX1' model file..." % len(X))
        Y_aux1 = aux1_model.predict(X, verbose = verbose)
        Y_aux1_columns = [node.op.name for node in aux1_model.outputs]

        if verbose: print("Predicting %d (x,y) coordinates for 'AUX2' model file..." % len(X))
        Y_aux2 = aux2_model.predict(X, verbose = verbose)
        Y_aux2_columns = [node.op.name for node in aux2_model.outputs]

        if verbose: print("Predictions completed!")

        return Y_main, Y_aux1, Y_aux2, Y_main_columns, Y_aux1_columns, Y_aux2_columns

    def get_keras_inception_v1_inspired(self, input_shape, return_model_only = True, include_head = False):
        kernel_init = glorot_uniform()
        bias_init = Constant(value = 0.2)
        input_img = Input(shape=input_shape)

        # Top Layer (Begin MODEL)
        model = Convolution2D(filters = 512, kernel_size = (3, 3), padding = 'same', strides = (2, 1),
            activation = 'relu', name = 'conv_1_7x7/2', kernel_initializer = kernel_init,
            bias_initializer = bias_init) (input_img)
        model = MaxPooling2D((3, 3), padding = 'same', strides = (1, 1), name = 'max_pool_1_3x3/2') (model)
        model = Convolution2D(256, (1, 1), padding = 'same', activation = 'relu', name = 'conv_2a_3x3/1') (model)
        model = Convolution2D(128, (3, 3), padding = 'same', activation = 'relu', name = 'conv_2b_3x3/1') (model)
        model = MaxPooling2D((3, 3), padding = 'same', strides = (2, 1), name = 'max_pool_2_3x3/2') (model)

        # Inception Module
        model = self.__inception_module(model,
            filters_1x1 = 64,
            filters_3x3_reduce = 96,
            filters_3x3 = 128,
            filters_5x5_reduce = 16,
            filters_5x5 = 32,
            filters_pool_proj = 32,
            kernel_init = kernel_init,
            bias_init = bias_init,
            name = 'inception_3a')

        # Inception Module
        model = self.__inception_module(model,
            filters_1x1 = 128,
            filters_3x3_reduce = 128,
            filters_3x3 = 192,
            filters_5x5_reduce = 32,
            filters_5x5 = 96,
            filters_pool_proj = 64,
            kernel_init = kernel_init,
            bias_init = bias_init,
            name = 'inception_3b')

        model = MaxPooling2D((3, 3), padding = 'same', strides = (3, 1), name= 'max_pool_3_3x3/2') (model)

        # Inception Module
        model = self.__inception_module(model,
            filters_1x1 = 192,
            filters_3x3_reduce = 96,
            filters_3x3 = 208,
            filters_5x5_reduce = 16,
            filters_5x5 = 48,
            filters_pool_proj = 64,
            kernel_init = kernel_init,
            bias_init = bias_init,
            name = 'inception_4a')

        # CDB 3/5 DROPOUT ADDED
        model = Dropout(0.2) (model)

        # Begin MODEL1 (auxillary output)
        #Our shape is already too small in the encoder dimension
        #model1 = AveragePooling2D((5, 5), padding = 'same', strides = 3, name= 'avg_pool_4_5x5/2') (model)

        model1 = Convolution2D(128, (1, 1), padding = 'same', activation = 'relu') (model)

        if include_head:
            model1 = Flatten() (model1)
            model1 = Dense(1024, activation = 'relu') (model1)
            model1 = Dropout(0.3) (model1)
            model1 = Dense(30, name = 'auxilliary_output_1') (model1)

        # Resume MODEL w/ Inception
        model = self.__inception_module(model,
            filters_1x1 = 160,
            filters_3x3_reduce = 112,
            filters_3x3 = 224,
            filters_5x5_reduce = 24,
            filters_5x5 = 64,
            filters_pool_proj = 64,
            kernel_init = kernel_init,
            bias_init = bias_init,
            name='inception_4b')

        model = self.__inception_module(model,
            filters_1x1 = 128,
            filters_3x3_reduce = 128,
            filters_3x3 = 256,
            filters_5x5_reduce = 24,
            filters_5x5 = 64,
            filters_pool_proj = 64,
            kernel_init = kernel_init,
            bias_init = bias_init,
            name='inception_4c')

        model = self.__inception_module(model,
            filters_1x1 = 112,
            filters_3x3_reduce = 144,
            filters_3x3 = 288,
            filters_5x5_reduce = 32,
            filters_5x5 = 64,
            filters_pool_proj = 64,
            kernel_init = kernel_init,
            bias_init = bias_init,
            name='inception_4d')

        # CDB : 3/5 DROPOUT ADDED
        model = Dropout(0.2) (model)

        # Begin MODEL2 (auxilliary output)

        #Our shape is already too small in the encoder dimension
        #model2 = AveragePooling2D((5, 5), strides = 3) (model)
        model2 = Convolution2D(128, (1, 1), padding = 'same', activation = 'relu') (model)

        if include_head:
            model2 = Flatten() (model2)
            model2 = Dense(1024, activation = 'relu') (model2)
            model2 = Dropout(0.3) (model2)
            model2 = Dense(30, name = 'auxilliary_output_2') (model2)

        # Resume MODEL w/ Inception
        model = self.__inception_module(model,
            filters_1x1 = 256,
            filters_3x3_reduce = 160,
            filters_3x3 = 320,
            filters_5x5_reduce = 32,
            filters_5x5 = 128,
            filters_pool_proj = 128,
            kernel_init = kernel_init,
            bias_init = bias_init,
            name='inception_4e')

        model = MaxPooling2D((3, 3), padding = 'same', strides = (2, 1), name = 'max_pool_4_3x3/2') (model)

        model = self.__inception_module(model,
            filters_1x1 = 256,
            filters_3x3_reduce = 160,
            filters_3x3 = 320,
            filters_5x5_reduce = 32,
            filters_5x5 = 128,
            filters_pool_proj = 128,
            kernel_init = kernel_init,
            bias_init = bias_init,
            name='inception_5a')

        model = self.__inception_module(model,
            filters_1x1 = 384,
            filters_3x3_reduce = 192,
            filters_3x3 = 384,
            filters_5x5_reduce = 48,
            filters_5x5 = 128,
            filters_pool_proj = 128,
            kernel_init = kernel_init,
            bias_init = bias_init,
            name='inception_5b')

        if include_head:
            # Output Layer (Main)
            model = GlobalAveragePooling2D(name = 'avg_pool_5_3x3/1') (model)
            model = Dropout(0.3) (model)
            model = Dense(30, name = 'main_output') (model)

        model = Model(input_img, [model, model1, model2], name = 'Inception')

        if return_model_only:
            return model
