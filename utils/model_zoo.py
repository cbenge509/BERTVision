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

from keras.layers.advanced_activations import LeakyReLU, ELU, ReLU
from keras.models import Sequential, Model, model_from_json
from keras.layers import Activation, Convolution2D, Conv2D, LocallyConnected2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, 
from keras.layers import BatchNormalization, Flatten, Dense, Dropout, Input, concatenate, add, Add, ZeroPadding2D, GlobalMaxPooling2D, DepthwiseConv2D
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l2
#from keras.activations import linear, elu, tanh, relu
from keras import metrics, losses, initializers, backend
from keras.losses import SparseCategoricalCrossentropy
from keras.utils import multi_gpu_model
from keras.initializers import glorot_uniform, Constant, lecun_uniform
from keras import backend as K

#os.environ["PATH"] += os.pathsep + "C:/ProgramData/Anaconda3/GraphViz/bin/"
#os.environ["PATH"] += os.pathsep + "C:/Anaconda/Graphviz2.38/bin/"

from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.python.keras import backend, initializers, models, regularizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
layers = tf.keras.layers
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
        self.__model_tasks = ['QnA', 'binary_classification']

    ######################################################
    ######################################################
    ######################################################
    ### Private Methods
    ######################################################
    ######################################################
    ######################################################

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
    
    # BERT "image" layer dedimensionalization
    def __BERT_image_input_layer(self, input_img, use_l2_regularizer, input_shape = (386, 1024, 3),  verbose = False):
        
        x = layers.Conv2D(
            filters = 3, 
            kernel_size = (1, 3),
            strides = (1, 3),
            padding = 'valid',
            use_bias = False,
            data_format = 'channels_last',
            kernel_initializer = 'he_normal',
            kernel_regularizer = self.__gen_l2_regularizer(use_l2_regularizer),
            name='input_conv') (input_img)
        
        return x

    #/////////////////////////////////////////////////////
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    ### ResNet v1.5 Private Methods
    #/////////////////////////////////////////////////////
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    def __gen_l2_regularizer(self, use_l2_regularizer = True, l2_weight_decay=1e-4):
        return regularizers.l2(l2_weight_decay) if use_l2_regularizer else None


    def __identity_block(self, input_tensor,
                    kernel_size,
                    filters,
                    stage,
                    block,
                    use_l2_regularizer=True,
                    batch_norm_decay=0.9,
                    batch_norm_epsilon=1e-5):
        """The identity block is the block that has no conv layer at shortcut.
        Args:
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            use_l2_regularizer: whether to use L2 regularizer on Conv layer.
            batch_norm_decay: Moment of batch norm layers.
            batch_norm_epsilon: Epsilon of batch borm layers.
        Returns:
            Output tensor for the block.
        """

        filters1, filters2, filters3 = filters
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = layers.Conv2D(
                filters1, (1, 1),
                use_bias=False,
                kernel_initializer = 'he_normal',
                kernel_regularizer = self.__gen_l2_regularizer(use_l2_regularizer),
                name=conv_name_base + '2a') (input_tensor)
        
        x = layers.BatchNormalization(
                axis=bn_axis,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                name=bn_name_base + '2a') (x)
        
        x = layers.Activation('relu') (x)

        x = layers.Conv2D(
                filters2,
                kernel_size,
                padding='same',
                use_bias=False,
                kernel_initializer = 'he_normal',
                kernel_regularizer = self.__gen_l2_regularizer(use_l2_regularizer),
                name=conv_name_base + '2b') (x)
        
        x = layers.BatchNormalization(
                axis=bn_axis,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                name=bn_name_base + '2b') (x)
        
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(
                filters3, (1, 1),
                use_bias=False,
                kernel_initializer = 'he_normal',
                kernel_regularizer = self.__gen_l2_regularizer(use_l2_regularizer),
                name=conv_name_base + '2c') (x)
        
        x = layers.BatchNormalization(
                axis=bn_axis,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                name=bn_name_base + '2c') (x)

        x = layers.add([x, input_tensor])
        x = layers.Activation('relu') (x)

        return x

    def __conv_block(self, input_tensor,
                kernel_size,
                filters,
                stage,
                block,
                strides = (2, 2),
                use_l2_regularizer = True,
                batch_norm_decay = 0.9,
                batch_norm_epsilon = 1e-5):
        """A block that has a conv layer at shortcut.
        Note that from stage 3,
        the second conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well
        Args:
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Strides for the second conv layer in the block.
            use_l2_regularizer: whether to use L2 regularizer on Conv layer.
            batch_norm_decay: Moment of batch norm layers.
            batch_norm_epsilon: Epsilon of batch borm layers.
        Returns:
            Output tensor for the block.
        """

        filters1, filters2, filters3 = filters
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = layers.Conv2D(
                filters1, (1, 1),
                use_bias=False,
                kernel_initializer = 'he_normal',
                kernel_regularizer = self.__gen_l2_regularizer(use_l2_regularizer),
                name=conv_name_base + '2a') (input_tensor)
        
        x = layers.BatchNormalization(
                axis=bn_axis,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                name=bn_name_base + '2a')(x)
        
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(
                filters2,
                kernel_size,
                strides=strides,
                padding='same',
                use_bias=False,
                kernel_initializer = 'he_normal',
                kernel_regularizer = self.__gen_l2_regularizer(use_l2_regularizer),
                name=conv_name_base + '2b') (x)
        
        x = layers.BatchNormalization(
                axis=bn_axis,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                name=bn_name_base + '2b') (x)
        
        x = layers.Activation('relu') (x)

        x = layers.Conv2D(
                filters3, (1, 1),
                use_bias=False,
                kernel_initializer = 'he_normal',
                kernel_regularizer = self.__gen_l2_regularizer(use_l2_regularizer),
                name=conv_name_base + '2c') (x)

        x = layers.BatchNormalization(
                axis=bn_axis,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                name=bn_name_base + '2c') (x)

        shortcut = layers.Conv2D(
                filters3, (1, 1),
                strides=strides,
                use_bias=False,
                kernel_initializer = 'he_normal',
                kernel_regularizer = self.__gen_l2_regularizer(use_l2_regularizer),
                name=conv_name_base + '1')(input_tensor)
        
        shortcut = layers.BatchNormalization(
                axis=bn_axis,
                momentum=batch_norm_decay,
                epsilon=batch_norm_epsilon,
                name=bn_name_base + '1') (shortcut)

        x = layers.add([x, shortcut])
        x = layers.Activation('relu') (x)

        return x

    ######################################################
    ######################################################
    ######################################################
    ### MODEL ZOO
    ######################################################
    ######################################################
    ######################################################



    #/////////////////////////////////////////////////////
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    ### ResNet v1.5 Model (TensorFlow implementation)
    #/////////////////////////////////////////////////////
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # slightly modified variant from tf model garden : https://github.com/tensorflow/models/blob/master/official/vision/image_classification/resnet/resnet_model.py

    def get_resnet50_v1_5(self, X, Y, batch_size, epoch_count, val_split = 0.1, shuffle = True, 
            recalculate_pickle = True, X_val = None, Y_val = None, task = "QnA", use_l2_regularizer = True, 
            batch_norm_decay = 0.9, batch_norm_epsilon = 1e-5, verbose = False):
        """Trains and returns a ResNet50 v1.5 Model
        Args:
            X (np.array): Input training data (images in HxWxC format)
            Y (np.array): [description]
            batch_size (int): Size of the batches for each step.
            epoch_count ([type]): [description]
            val_split (float, optional): [description]. Defaults to 0.1.
            shuffle (bool, optional): [description]. Defaults to True.
            recalculate_pickle (bool, optional): [description]. Defaults to True.
            X_val ([type], optional): [description]. Defaults to None.
            Y_val ([type], optional): [description]. Defaults to None.
            task (str, optional): [description]. Defaults to "QnA".
            use_l2_regularizer (bool, optional): whether to use L2 regularizer on Conv/Dense layer. Defaults to True.
            batch_norm_decay (float, optional): Moment of batch norm layers. Defaults to 0.9.
            batch_norm_epsilon ([type], optional): Epsilon of batch borm layers. Defaults to 1e-5.
            verbose (bool, optional): [description]. Defaults to False.

        Raises:
            RuntimeError: If the 'task' parameter specified is not 'binary_classification' or 'QnA'

        Returns:
            [tf.keras.Model]: a trained Keras ResNet50 v1.5 Model object
        """

        if task not in self.__model_tasks:
            raise RuntimeError(f"parameter task value of '{task}' is not permitted.")

        __MODEL_NAME = "ResNet50_v1_5"
        __MODEL_NAME_TASK = "".join([__MODEL_NAME, "_", task])
        __MODEL_FNAME_PREFIX = "ResNet50_v1_5/"
        
        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            os.makedirs(nested_dir)

        __model_file_name = "".join([nested_dir, __MODEL_NAME_TASK, ".h5"])
        __model_json_file = "".join([nested_dir, __MODEL_NAME_TASK, ".json"])
        __model_architecture_plot_file = "".join([nested_dir, __MODEL_NAME_TASK, "_plot.png"])
        __history_params_file = "".join([nested_dir, __MODEL_NAME_TASK, "_params.csv"])
        __history_performance_file = "".join([nested_dir, __MODEL_NAME_TASK, "_history.csv"])
        __history_plot_file = "".join([nested_dir, __MODEL_NAME_TASK, "_output_plot.png"])

        if verbose: print(f"Retrieving model: {__MODEL_NAME}...")

        # Create or load the model
        if (not os.path.isfile(__model_file_name)) or (not os.path.isfile(__model_json_file)) or recalculate_pickle:
            if verbose: print(f"Pickle file for {__MODEL_NAME} and task {task} MODEL not found or skipped by caller.")

            opt = Adam(lr = 1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
            
            if task == "binary_classification":
                mtrc = ['accuracy']
                cp = ModelCheckpoint(filepath = __model_file_name, verbose = verbose, save_best_only = True, 
                    mode = 'min', monitor = 'val_accuracy')
                lss = SparseCategoricalCrossentropy(from_logits = True)
            elif task == "QnA":
                mtrc = ['mse']
                cp_main = ModelCheckpoint(filepath = __model_file_name, verbose = verbose, save_best_only = True, 
                    mode = 'min', monitor = 'val_mse')
                lss = 'mean_squared_error'
            
            stop_at = np.max([int(0.1 * epoch_count), self.__MIN_early_stopping])
            es = EarlyStopping(patience = stop_at, verbose = verbose)

            kernel_init = glorot_uniform()
            bias_init = Constant(value = 0.2)
            
            # channels_last
            bn_axis = 3
            block_config = dict(
                use_l2_regularizer = use_l2_regularizer,
                batch_norm_decay = batch_norm_decay,
                batch_norm_epsilon = batch_norm_epsilon)

            if self.__GPU_count > 1: dev = "/cpu:0"
            else: dev = "/gpu:0"
            with tf.device(dev):

                # input image size of 386h x 1024w x 3c
                input_img = layers.Input(shape = (386, 1024, 3), batch_size = batch_size)

                # downscale our 386x1024 images across the width dimension
                x = self.__BERT_image_input_layer(
                    input_img = input_img,
                    use_l2_regularizer = use_l2_regularizer, 
                    input_shape = (386, 1024, 3),
                    verbose = verbose)
                
                x = layers.ZeroPadding2D(padding = (3, 3), name = 'conv1_pad') (x)
                
                x = layers.Conv2D(
                    filters = 64, 
                    kernel_size = (7, 7),
                    strides = (2, 2),
                    padding = 'valid',
                    use_bias = False,
                    kernel_initializer = 'he_normal',
                    kernel_regularizer = self.__gen_l2_regularizer(use_l2_regularizer),
                    name = 'conv1') (x)
                
                x = layers.BatchNormalization(
                    axis = bn_axis,
                    momentum = batch_norm_decay,
                    epsilon = batch_norm_epsilon,
                    name = 'bn_conv1') (x)

                x = layers.Activation('relu') (x)
                x = layers.MaxPooling2D((3, 3), strides = (2, 2), padding = 'same') (x)

                x = self.__conv_block(input_tensor = x, kernel_size = 3, filters = [64, 64, 256], stage = 2, block = 'a', strides = (1, 1), **block_config)
                x = self.__identity_block(input_tensor = x, kernel_size = 3, filters = [64, 64, 256], stage = 2, block = 'b', **block_config)
                x = self.__identity_block(input_tensor = x, kernel_size = 3, filters = [64, 64, 256], stage = 2, block = 'c', **block_config)

                x = self.__conv_block(input_tensor = x, kernel_size = 3, filters = [128, 128, 512], stage = 3, block = 'a', **block_config)
                x = self.__identity_block(input_tensor = x, kernel_size = 3, filters = [128, 128, 512], stage = 3, block = 'b', **block_config)
                x = self.__identity_block(input_tensor = x, kernel_size = 3, filters = [128, 128, 512], stage = 3, block = 'c', **block_config)
                x = self.__identity_block(input_tensor = x, kernel_size = 3, filters = [128, 128, 512], stage = 3, block = 'd', **block_config)

                x = self.__conv_block(input_tensor = x, kernel_size = 3, filters = [256, 256, 1024], stage = 4, block = 'a', **block_config)
                x = self.__identity_block(input_tensor = x, kernel_size = 3, filters = [256, 256, 1024], stage = 4, block = 'b', **block_config)
                x = self.__identity_block(input_tensor = x, kernel_size = 3, filters = [256, 256, 1024], stage = 4, block = 'c', **block_config)
                x = self.__identity_block(input_tensor = x, kernel_size = 3, filters = [256, 256, 1024], stage = 4, block = 'd', **block_config)
                x = self.__identity_block(input_tensor = x, kernel_size = 3, filters = [256, 256, 1024], stage = 4, block = 'e', **block_config)
                x = self.__identity_block(input_tensor = x, kernel_size = 3, filters = [256, 256, 1024], stage = 4, block = 'f', **block_config)

                x = self.__conv_block(input_tensor = x, kernel_size = 3, filters = [512, 512, 2048], stage = 5, block = 'a', **block_config)
                x = self.__identity_block(input_tensor = x, kernel_size = 3, filters = [512, 512, 2048], stage = 5, block = 'b', **block_config)
                x = self.__identity_block(input_tensor = x, kernel_size = 3, filters = [512, 512, 2048], stage = 5, block = 'c', **block_config)

                x = layers.GlobalAveragePooling2D() (x)
                
                x = layers.Dense(2,
                    kernel_initializer = initializers.RandomNormal(stddev = 0.01),
                    kernel_regularizer = self.__gen_l2_regularizer(use_l2_regularizer),
                    bias_regularizer = self.__gen_l2_regularizer(use_l2_regularizer),
                    name = 'fc1000') (x)

                if task == "binary_classification":
                    # A softmax that is followed by the model loss must be done cannot be done
                    # in float16 due to numeric issues. So we pass dtype=float32.
                    x = layers.Activation('softmax', dtype = 'float32') (x)
                elif task == "QnA":
                    x = layers.Activation('relu') (x)

                model = models.Model(input_img, x, name = 'ResNet50_v1_5')

            if verbose: print(model.summary())

            # Compile the model
            if self.__GPU_count > 1:
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    parallel_model = multi_gpu_model(model, gpus = self.__GPU_count)
                    parallel_model.compile(optimizer = opt, loss = lss, metrics = mtrc)
            else:
                parallel_model = model
                parallel_model.compile(optimizer = opt, loss = lss, metrics = mtrc)

            if (X_val is None) or (Y_val is None):
                history = parallel_model.fit(X, Y, validation_split = val_split, batch_size = batch_size * self.__GPU_count, 
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp_main], verbose = verbose)
            else:
                history = parallel_model.fit(X, Y, validation_data = (X_val, Y_val), batch_size = batch_size * self.__GPU_count, 
                    epochs = epoch_count, shuffle = shuffle, callbacks = [es, cp_main], verbose = verbose)

            # print and/or save a performance plot
            try:
                if task == "binary_classification":
                    self.__plot_keras_history(history = history, metric = 'accuracy', model_name = __MODEL_NAME, 
                        file_name = __history_plot_file, verbose = False)
                elif task == "QnA":
                    self.__plot_keras_history(history = history, metric = 'mse', model_name = __MODEL_NAME, 
                        file_name = __history_plot_file, verbose = False)
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

        if verbose: print(f"Loading pickle file for '{__MODEL_NAME}' model (task: {task}) from file '{__model_file_name}'")
        parallel_model = self.__load_keras_model(__MODEL_NAME, __model_file_name, __model_json_file, verbose = verbose)

        return parallel_model, hist_params, hist

    # ********************************
    # ***** ResNet50 v1.5 INFERENCING
    # ********************************
    def predict_resnet50_v1_5(self, X, task = "QnA", verbose = False):
        """Inferencing for ResNet50 v1.5

        Args:
            X ([np.array]): input vector for inferencing
            verbose (bool, optional): Verbose messaging to caller. Defaults to False.

        Raises:
            RuntimeError: [description]
            RuntimeError: [description]

        Returns:
            np.array: Predictions from ResNet50 v1.5
        """

        __MODEL_NAME = "ResNet50_v1_5"
        __MODEL_NAME_TASK = "".join([__MODEL_NAME, "_", task])
        __MODEL_FNAME_PREFIX = "ResNet50_v1_5/"
        
        nested_dir = "".join([self.__models_path,__MODEL_FNAME_PREFIX])
        if not os.path.exists(nested_dir):
            os.makedirs(nested_dir)

        __model_file_name = "".join([nested_dir, __MODEL_NAME_TASK, ".h5"])
        __model_json_file = "".join([nested_dir, __MODEL_NAME_TASK, ".json"])


        if (not os.path.isfile(__model_file_name)) or (not os.path.isfile(__model_json_file)):
            raise RuntimeError("One or some of the following files are missing; prediction cancelled:\n\n'%s'\n'%s'\n\n" % 
                (__model_file_name, __model_json_file))
        
        # load the Keras model for the specified feature
        model = self.__load_keras_model(__MODEL_NAME, __model_file_name, __model_json_file, verbose = verbose)

        # predict
        if verbose: print("Predicting %d instances..." % len(X))
        Y = model.predict(X, verbose = verbose)

        if verbose: print("Predictions completed!")

        return Y

