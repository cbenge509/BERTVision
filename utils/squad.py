############################################################################
# IMPORTS
############################################################################

import os
import h5py
import numpy as np
import json
from transformers import BertTokenizer

import tensorflow as tf
from transformers.data.processors.squad import SquadV2Processor
from transformers.data.processors.squad import squad_convert_examples_to_features

from transformers import TFBertForQuestionAnswering
from transformers import BertConfig

from collections import defaultdict, Counter

from transformers import BertTokenizer, TFBertModel, BertConfig
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

from utils.evaluation import Squad2Config

np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

############################################################################
# CONSTANTS & PARAMETERS
############################################################################

# filenames from the (input) SQuAD v2 dataset
SQUAD_DEV_FILE = "dev-v2.0.json"
SQUAD_TRAIN_FILE = "train-v2.0.json"
# filenames for the (output) h5 SQuAD v2 dev and train processed dataset
OUTPUT_DEV_FILE = "squad_dev.h5"
OUTPUT_TRAIN_FILE = "squad_train.h5"

############################################################################
# CLASS : SQuADv2Utils
############################################################################

class SQuADv2Utils(object):

    #---------------------------------------------------------------------------
    # CLass Initialization
    #---------------------------------------------------------------------------
    def __init__(self, data_path, h5_path, pretrained_tokenizer = "bert-large-uncased", verbose = False):
        """ProcessSquad class initialization routine.

        Args:
            data_path (str): OS path location of the SQuAD v2 dev and train files.
            h5_path (str): OS path location of the output folder where the h5 processed SQuAD data should be stored.
            pretrained_tokenizer (str): name of the pretrained tokenizer to use during processing (ref: https://huggingface.co/transformers/main_classes/tokenizer.html).
            verbose (bool, optional): Indicates whether the routine should provide verbose feedback to caller. Defaults to False.

        Raises:
            RuntimeError: if any path provided does not exist.
            RuntimeError: if the SQuAD v2 dev or training files do not exist in the data_path.
        """
        # validate that the constructor parameters were provided by caller
        if (not data_path) | (not h5_path) | (not pretrained_tokenizer):
            raise RuntimeError('SQuAD v2 data path, output h5 path, and pretrained_tokenizer must be specified.')

        # clean and validate the path strings
        data_path = self.__clean_path(data_path)
        h5_path = self.__clean_path(h5_path)

        # validate existence of the expected SQuAD v2 files in the data_path provided by caller
        for f, d in [[SQUAD_DEV_FILE, "SQuAD v2 Dev File"], [SQUAD_TRAIN_FILE, "SQuAD v2 Train File"]]:
            f = os.path.join(data_path, f)
            if (not os.path.isfile(f)):
                raise RuntimeError(f"{d} file specified [{f}] does not exist.")

        # set the class variables with the dev and train squad file locations
        self.__dev_squad = data_path
        self.__train_squad = data_path

        # set the class variable for the h5 output files
        self.__dev_h5 = os.path.join(h5_path, OUTPUT_DEV_FILE)
        self.__train_h5 = os.path.join(h5_path, OUTPUT_TRAIN_FILE)

        # load the pre-trained tokenizer
        pretrained_tokenizer = pretrained_tokenizer.strip().lower()
        try:
            self.__tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer)
        except:
            raise RuntimeError(f"Failed to load pretrained tokenizer '{pretrained_tokenizer}'.")

        # Load the processor
        self.__processor = SquadV2Processor()

        if verbose: print("All input file locations validated.")

    #---------------------------------------------------------------------------
    # (Utility) __clean_path
    #---------------------------------------------------------------------------
    def __clean_path(self, clean_path, path_title = "Unspecified"):
        """Common utility function for cleaning and validating path strings

        Args:
            clean_path (str): the path string to clean & validate
            path_title (str): the title (or name) of the path; used on error only.

        Returns:
            (str): the cleaned path string
        """

        clean_path = str(clean_path).replace('\\', '/').strip()
        if (not clean_path.endswith('/')): clean_path = ''.join((clean_path, '/'))

        if (not os.path.isdir(clean_path)):
            raise RuntimeError(f"'{path_title}' path specified [{clean_path}] is invalid.")

        return clean_path

    #---------------------------------------------------------------------------
    # (Public Method) GenerateFeatures
    #---------------------------------------------------------------------------
    def GenerateFeatures(self, generate_training = True, max_seq_length = 512, max_query_length = 64, doc_stride = 128, verbose = False):
        """Converts examples into features

        Args:
            generate_training (bool, optional): Indicates if the examples are from the training set. Defaults to True.
            max_seq_length (int, optional): Maximum sequence length. Defaults to 512.
            max_query_length (int, optional): Maximum query length. Defaults to 64.
            doc_stride (int, optional): Document stride vale. Defaults to 128.
            verbose (bool, optional): Enables verbose logging to console. Defaults to False.
        """

        if generate_training:
            feature_target = "train"
        else:
            feature_target = "dev"

        if verbose: print(f"Collecting the raw '{feature_target}' examples for processing.")
        if generate_training:
            data_raw = self.__processor.get_train_examples(self.__train_squad)
            data_h5 = self.__train_h5
        else:
            data_raw = self.__processor.get_dev_examples(self.__dev_squad)
            data_h5 = self.__dev_h5

        # ref: https://huggingface.co/transformers/main_classes/processors.html?highlight=squad_convert_examples_to_features#transformers.data.processors.squad.squad_convert_examples_to_features
        if verbose: print(f"Converting list of '{feature_target}' examples to list of features...")
        data = squad_convert_examples_to_features(
            examples = data_raw,
            tokenizer = self.__tokenizer,
            max_seq_length = max_seq_length,
            doc_stride = doc_stride,
            max_query_length = max_query_length,
            is_training = generate_training,)

        feature_length = len(data)
        if verbose: print(f"Conversion complete.  Length of {feature_target} dataset: {feature_length}")

        # create zero-initialized arrays for storing the featurized training set
        input_ids, input_segs, input_masks = np.zeros((3, feature_length, max_seq_length))
        input_start, input_end, input_is_impossible = np.zeros((3, feature_length))

        # populate zero-initialized arrays with features generated above
        arrz = [input_ids, input_segs, input_masks, input_start, input_end, input_is_impossible]
        #varz = ["input_ids", "token_type_ids", "attention_mask", "start_position", "end_position", "is_impossible"]
        varz = ["input_ids", "token_type_ids", "attention_mask", "input_start", "input_end", "input_is_impossible"]
        if verbose: print("generating arrays for binarization...")
        for i, d in enumerate(data):
            for a, v in zip(arrz, [d.input_ids, d.token_type_ids, d.attention_mask, d.start_position, d.end_position, d.is_impossible]):
                a[i] = v

        # save the h5 file
        if verbose: print(f"writing to {feature_target} file '{data_h5}'...")
        with h5py.File(data_h5, 'w') as hf:
            for a, v in zip(arrz, varz):
                hf.create_dataset(v, data = a)
        if verbose: print(f"{feature_target} H5 feature file '{data_h5}' written to disk.")

        return

class UntrainedBertSquad2Faster(object):
    def __init__(self,
                 config = Squad2Config()):

        self.tokenizer = config.tokenizer
        self.named_model = config.named_model
        self.model = self.bert_large_uncased_for_squad2(config.max_seq_length)

    def bert_large_uncased_for_squad2(self, max_seq_length):
        input_ids = Input((max_seq_length,), dtype = tf.int32, name = 'input_ids')
        input_masks = Input((max_seq_length,), dtype = tf.int32, name = 'input_masks')
        input_tokens = Input((max_seq_length,), dtype = tf.int32, name = 'input_tokens')

        #Load model from huggingface
        config = BertConfig.from_pretrained("bert-large-uncased", output_hidden_states=True)
        bert_layer = TFBertModel.from_pretrained(self.named_model, config = config)
        bert_layer.load_weights('bert_base_squad_1e-5_adam_4batchsize_4epochs_weights_BERT_ONLY.h5')
        _, _, embeddings = bert_layer([input_ids, input_masks, input_tokens]) #1 for pooled outputs, 0 for sequence

        model = Model(inputs = [input_ids, input_masks, input_tokens], outputs = embeddings)
        return model

class UntrainedBertSquad2(object):
    def __init__(self,
                 config = Squad2Config()):

        self.tokenizer = config.tokenizer
        self.named_model = config.named_model
        self.model = self.bert_large_uncased_for_squad2(config.max_seq_length)

    def bert_large_uncased_for_squad2(self, max_seq_length):
        input_ids = Input((max_seq_length,), dtype = tf.int32, name = 'input_ids')
        input_masks = Input((max_seq_length,), dtype = tf.int32, name = 'input_masks')

        #Load model from huggingface
        config = BertConfig.from_pretrained(self.named_model, output_hidden_states=True)
        bert_layer = TFBertModel.from_pretrained(self.named_model, config = config)

        outputs, _, embeddings = bert_layer([input_ids, input_masks]) #1 for pooled outputs, 0 for sequence

        model = Model(inputs = [input_ids, input_masks], outputs = [embeddings, outputs])
        return model

class FineTunedBertSquad2(object):
    def __init__(self, weights_file = None,
                 config = Squad2Config()):

        self.tokenizer = config.tokenizer
        self.named_model = config.named_model
        self.weights_file = weights_file
        self.model = self.bert_large_uncased_for_squad2(config.max_seq_length)

    def bert_large_uncased_for_squad2(self, max_seq_length):
        input_ids = Input((max_seq_length,), dtype = tf.int32, name = 'input_ids')
        input_masks = Input((max_seq_length,), dtype = tf.int32, name = 'input_masks')
        input_tokens = Input((max_seq_length,), dtype = tf.int32, name = 'input_tokens')

        #Load model from huggingface
        config = BertConfig.from_pretrained("bert-large-uncased", output_hidden_states=True)
        bert_layer = TFBertModel.from_pretrained(self.named_model, config = config)
        if self.weights_file is not None:
            bert_layer.load_weights(self.weights_file)
        _, _, embeddings = bert_layer([input_ids, input_masks, input_tokens]) #1 for pooled outputs, 0 for sequence

        model = Model(inputs = [input_ids, input_masks, input_tokens], outputs = embeddings)
        return model
