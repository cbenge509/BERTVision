#%%
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

np.random.seed(42)
tf.random.set_seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

#%%
############################################################################
# CONSTANTS & PARAMETERS
############################################################################

# filenames from the (input) SQuAD v2 dataset
SQUAD_DEV_FILE = "dev-v2.0.json"
SQUAD_TRAIN_FILE = "train-v2.0.json"
# filenames for the (output) h5 SQuAD v2 dev and train processed dataset
OUTPUT_DEV_FILE = "squad_dev.h5"
OUTPUT_TRAIN_FILE = "squad_train.h5"

#%%
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
            data_path ([str]): OS path location of the SQuAD v2 dev and train files.
            h5_path ([str]): OS path location of the output folder where the h5 processed SQuAD data should be stored.
            pretrained_tokenizer ([str]): name of the pretrained tokenizer to use during processing (ref: https://huggingface.co/transformers/main_classes/tokenizer.html).
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
                raise RuntimeError("%s file specified [%s] does not exist." % (d, f))
        
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
            raise RuntimeError("Failed to load pretrained tokenizer '%s'." % pretrained_tokenizer)

        # Load the processor
        self.__processor = SquadV2Processor()

        if verbose: print("All input file locations validated.")

    #---------------------------------------------------------------------------
    # (Utility) __clean_path
    #---------------------------------------------------------------------------
    def __clean_path(self, clean_path, path_title = "Unspecified"):
        """Common utility function for cleaning and validating path strings

        Args:
            clean_path ([str]): the path string to clean & validate
            path_title ([str]): the title (or name) of the path; used on error only.

        Returns:
            [str]: the cleaned path string
        """

        clean_path = str(clean_path).replace('\\', '/').strip()
        if (not clean_path.endswith('/')): clean_path = ''.join((clean_path, '/'))

        if (not os.path.isdir(clean_path)):
            raise RuntimeError("'%s' path specified '%s' is invalid." % (path_title, clean_path))

        return clean_path

    #---------------------------------------------------------------------------
    # (Public Method) GenerateFeatures
    #---------------------------------------------------------------------------
    def GenerateFeatures(self, generate_training = True, max_seq_length = 512, max_query_length = 64, doc_stride = 128, verbose = False):

        if generate_training:
            feature_target = "train"
        else:
            feature_target = "dev"
        
        if verbose: print("Collecting the raw '%s' examples for processing." % feature_target)
        if generate_training:
            data_raw = self.__processor.get_train_examples(self.__train_squad)
            data_h5 = self.__dev_h5
        else:
            data_raw = self.__processor.get_dev_examples(self.__dev_squad)
            data_h5 = self.__train_h5
        
        # ref: https://huggingface.co/transformers/main_classes/processors.html?highlight=squad_convert_examples_to_features#transformers.data.processors.squad.squad_convert_examples_to_features
        if verbose: print("Converting list of '%s' examples to list of features..." % feature_target)
        data = squad_convert_examples_to_features(
            examples = data_raw, 
            tokenizer = self.__tokenizer, 
            max_seq_length = max_seq_length, 
            doc_stride = doc_stride,
            max_query_length = max_query_length,
            is_training = generate_training,)
        
        feature_length = len(data)
        if verbose: print("Conversion complete.  Length of %s dataset: %d" % (feature_target, feature_length))
        
        # create zero-initialized arrays for storing the featurized training set
        input_ids, input_segs, input_masks = [np.zeros((feature_length, max_seq_length))] * 3
        input_start, input_end, input_is_impossible = [np.zeros((feature_length,))] * 3

        # populate zero-initialized arrays with features generated above
        arrz = [input_ids, input_segs, input_masks, input_start, input_end, input_is_impossible]
        varz = ["input_ids", "token_type_ids", "attention_mask", "start_position", "end_position", "is_impossible"]
        if verbose: print("generating arrays for binarization...")
        for i, d in enumerate(data):
            for a, v in zip(arrz, [d.input_ids, d.token_type_ids, d.attention_mask, d.start_position, d.end_position, d.is_impossible]):
                a[i] = v

        # save the h5 file
        if verbose: print("writing to %s file '%s'..." % (feature_target, data_h5))
        with h5py.File(data_h5, 'w') as hf:
            for a, v in zip(arrz, varz):
                hf.create_dataset(v, data = a)
        if verbose: print("%s H5 feature file '%s' written to disk." % (feature_target, data_h5))

        return
        