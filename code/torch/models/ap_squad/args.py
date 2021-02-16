import os, sys
sys.path.append("C:\\BERTVision\\code\\torch")
import torch
import models.args

def get_args():
    # retreive the general models.args and attach them here
    parser = models.args.get_args()

    # set some AP_SQuAD specific args
    parser.add_argument('--model',
                        type=str,
                        default='AP_SQuAD',
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
                        default=384,
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
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='Input batch size for training (default: 16)')
    parser.add_argument('--lr',
                        type=float,
                        default=2e-5,
                        help='Learning rate (default: 2e-5)')
    parser.add_argument('--num-workers',
                        type=int,
                        default=0,
                        help='H5 cannot be pickled')
    args = parser.parse_args()

    return args

#
