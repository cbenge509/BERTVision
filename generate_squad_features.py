############################################################################
# IMPORTS
############################################################################

import numpy as np
from utils import squad
import argparse
import os

############################################################################
# CONSTANTS & PARAMETERS
############################################################################

# Default file Locations (parameters)
SQUAD_PATH = "data/"
H5_PATH = "data/"

# Processing behavior (parameters)
PRETRAINED_TOKENIZER = "bert-large-uncased"
MAX_SEQUENCE_LENGTH = 512
MAX_QUERY_LENGTH = 64
DOCUMENT_STRIDE = 128
GENERATE_TARGET = "all"
VERBOSE = True

############################################################################
# ARGUMENT SPECIFICATION
############################################################################
parser = argparse.ArgumentParser(description = "Generates SQuAD v2 features from raw data and stores them in binary files.")
# Commandline arguments
parser.add_argument('-nv', '--no_verbose', action = 'store_true', help = 'Disables verbose output mode for more detailed descriptions of process.')
parser.add_argument('-tok', '--pretrained_tokenizer', type = str, default = PRETRAINED_TOKENIZER, help = "Specifies the pretrained tokenizer to use during training.")
parser.add_argument('-sfp', '--squad_file_path', type = str, default = SQUAD_PATH, help = "Path to location of SQuAD v2 raw dev and training JSON files.")
parser.add_argument('-h5p', '--h5_file_path', type = str, default = H5_PATH, help = "Path to store processed SQuAD v2 dev and training examples to.")
parser.add_argument('-sl', '--max_sequence_length', type = int, default = MAX_SEQUENCE_LENGTH, help = "Maximum sequence length to use in training.")
parser.add_argument('-ql', '--max_query_length', type = int, default = MAX_QUERY_LENGTH, help = "Maximum query length to use in training.")
parser.add_argument('-ds', '--document_stride', type = int, default = DOCUMENT_STRIDE, help = "Document stride to use in training.")
parser.add_argument('-t', '--target_to_generate', type = str, default = GENERATE_TARGET, help = 'Specifies which data to process; valid values are "train", "dev", or "all".')

############################################################################
# ARGUMENT PARSING
############################################################################
def process_arguments(parsed_args, display_args = False):
    """Command-line argument parsing

    Args:
        parsed_args (list): contains the list of arguments parsed by the command-line
        display_args (bool, optional): Enables verbose logging of the in-use command-line parameters to the console. Defaults to False.

    Raises:
        RuntimeError: if any of the path strings provided do not exist.
        RuntimeError: `pretrained_tokenizer` parameter value provided by caller is of the wrong data type.
        RuntimeError: `pretrained_tokenizer` parameter value provided by caller is in an invalid range.
        RuntimeError: `target_to_generate` parameter value provided by caller is of the wrong data type.
        RuntimeError: `target_to_generate` parameter value provided by caller is in an invalid range.
        RuntimeError: `max_sequence_length` parameter value provided by caller is of the wrong data type.
        RuntimeError: `max_sequence_length` parameter value provided by caller is in an invalid range.
        RuntimeError: `max_query_length` parameter value provided by caller is of the wrong data type.
        RuntimeError: `max_query_length` parameter value provided by caller is in an invalid range.
        RuntimeError: `document_stride` parameter value provided by caller is of the wrong data type.
        RuntimeError: `document_stride` parameter value provided by caller is in an invalid range.
    """
    
    global VERBOSE, SQUAD_PATH, H5_PATH, PRETRAINED_TOKENIZER, MAX_SEQUENCE_LENGTH, MAX_QUERY_LENGTH, \
           DOCUMENT_STRIDE, GENERATE_TARGET
    
    args = vars(parser.parse_args())
    if display_args:
        print("".join(["*" * 30, "\narguments in use:\n", "*" * 30, "\n"]))
        for arg in args:
            print("Parameter '%s' == %s" % (arg, str(getattr(parser.parse_args(), arg))))
        print("\n")

    # Assign arguments to globals
    VERBOSE = not args['no_verbose']
    SQUAD_PATH = args['squad_file_path'].strip().lower().replace('\\', '/')
    H5_PATH = args['h5_file_path'].strip().lower().replace('\\', '/')
    PRETRAINED_TOKENIZER = args['pretrained_tokenizer'].strip().lower()
    MAX_SEQUENCE_LENGTH = args['max_sequence_length']
    MAX_QUERY_LENGTH = args['max_query_length']
    DOCUMENT_STRIDE = args['document_stride']
    GENERATE_TARGET = args['target_to_generate'].strip().lower()
    
    # validate the existence of the caller-specified paths
    for p, v, l in zip([SQUAD_PATH, H5_PATH], ['squad_file_path', 'h5_file_path'], ['SQuAD v2 raw data path', 'H5 output path']):
        if not os.path.exists(p):
            raise RuntimeError(" ".join([l, "'%s'" % p, "specified in parameter `%s` does not exist." % v]))

    # validate type and value of `pretrained_tokenizer` parameter
    if not (isinstance(PRETRAINED_TOKENIZER, str)): raise RuntimeError("Parameter 'pretrained_tokenizer' must be of type str.")
    if not (0 < len(PRETRAINED_TOKENIZER) < 50): raise RuntimeError("Parameter 'pretrained_tokenizer' must be of a length between 1 and 50.")
    
    # validate type and value of `target_to_generate` parameter
    if not (isinstance(GENERATE_TARGET, str)): raise RuntimeError("Parameter 'target_to_generate' must be of type str.")
    if not (GENERATE_TARGET in ["train", "dev", "all"]): raise RuntimeError("Parameter 'target_to_generate' must be one of the following: {'train', 'dev', 'all'}.")

    # validate type and value of `max_sequence_length` parameter
    if not (isinstance(MAX_SEQUENCE_LENGTH, int)): raise RuntimeError("Parameter 'max_sequence_length' must be of type int.")
    if not (1 <= MAX_SEQUENCE_LENGTH <= 512): raise RuntimeError("Parameter 'max_sequence_length' must be between 1 and 512.")

    # validate type and value of `max_query_length` parameter
    if not (isinstance(MAX_QUERY_LENGTH, int)): raise RuntimeError("Parameter 'max_query_length' must be of type int.")
    if not (1 <= MAX_QUERY_LENGTH <= 128): raise RuntimeError("Parameter 'max_query_length' must be between 1 and 128.")

    # validate type and value of `document_stride` parameter
    if not (isinstance(DOCUMENT_STRIDE, int)): raise RuntimeError("Parameter 'document_stride' must be of type int.")
    if not (1 <= DOCUMENT_STRIDE <= 512): raise RuntimeError("Parameter 'document_stride' must be between 1 and 512.")

############################################################################
# PROCESS SQUAD V2 DATA
############################################################################

# process the SQuAD v2 data
def processSquad(processor, generate_training, max_seq_length, max_query_length, doc_stride, verbose = False):
    """Generates SQuAD features from the provided examples

    Args:
        processor (squad.SQuADv2Utils): class object; reference to the utilities helper class for processing SQuAD data.
        generate_training (bool): Indicates if the dataset consists of training examples.
        max_seq_length (int): Maximum sequence length.
        max_query_length (int): Maximum query length.
        doc_stride (int): Document stride.
        verbose (bool, optional): Enables verbose logging to the console. Defaults to False.
    """

    processor.GenerateFeatures(generate_training = generate_training, max_seq_length = max_seq_length,
        max_query_length = max_query_length, doc_stride = doc_stride, verbose = verbose)

    return

############################################################################
# MAIN FUNCTION
############################################################################

if __name__ == "__main__":

    # Clear the screen
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')
    
    # Process command-line arguments and set parameters
    process_arguments(parser.parse_args(), display_args = True)

    print("".join(["-" * 100, "\n>>> SQUAD V2 FEATURE GENERATION INITIATED <<<\n", "-" * 100, "\n"]))

    # Genrate the feature data and store as binary
    processor = squad.SQuADv2Utils(data_path = SQUAD_PATH, h5_path = H5_PATH,
        pretrained_tokenizer = PRETRAINED_TOKENIZER, verbose = VERBOSE)
    
    if GENERATE_TARGET in ["train", "all"]:
        if VERBOSE: print("".join(["\n", "=" * 50, "".join(["\nGenerating TRAIN data\n"]), "=" * 50, "\n"]))
        processSquad(processor = processor, generate_training = True, max_seq_length = MAX_SEQUENCE_LENGTH,
            max_query_length = MAX_QUERY_LENGTH, doc_stride = DOCUMENT_STRIDE, verbose = VERBOSE)

    if GENERATE_TARGET in ["dev", "all"]:
        if VERBOSE: print("".join(["\n", "=" * 50, "".join(["\nGenerating DEV data\n"]), "=" * 50, "\n"]))
        processSquad(processor = processor, generate_training = False, max_seq_length = MAX_SEQUENCE_LENGTH,
            max_query_length = MAX_QUERY_LENGTH, doc_stride = DOCUMENT_STRIDE, verbose = VERBOSE)

    print("".join(["-" * 100, "\n>>> SQUAD V2 FEATURE GENERATION COMPLETE <<<\n", "-" * 100, "\n"]))

