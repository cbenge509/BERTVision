import os, sys
sys.path.append("C:\\BERTVision\\code\\torch")
import torch
import models.args

def get_args():
    # retreive the general models.args and attach them here
    parser = models.args.get_args()

    # set some SST specific args
    parser.add_argument('--model',
                        type=str,
                        default='SST',
                        required=True)
    parser.add_argument('--checkpoint',
                        type=str,
                        default='bert-base-uncased',
                        required=True,
                        help='A HuggingFace checkpoint e.g., bert-base-uncased')
    parser.add_argument('--num-labels',
                        default=2,
                        type=int)
    parser.add_argument('--max-seq-length',
                        default=64,
                        type=int,
                        help='Tokenization max length')
    parser.add_argument('--save-path',
                        type=str,
                        default=os.path.join('model_checkpoints'))
    parser.add_argument('--log-path',
                        type=str,
                        default=os.path.join('model_logs'))
    parser.add_argument('--warmup-proportion',
                        default=0.1,
                        type=float,
                        help='Proportion of training to perform linear learning rate warmup for')
    args = parser.parse_args()

    return args

#
