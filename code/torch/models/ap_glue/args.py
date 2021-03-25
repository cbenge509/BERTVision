import os
import sys
sys.path.append("C:\\BERTVision")
import torch
import models.args


def get_args():
    # retreive the general models.args and attach them here
    parser = models.args.get_args()

    # set some AdapterPooler specific args
    parser.add_argument('--model',
                        type=str,
                        default='AP_RTE',
                        required=True)
    parser.add_argument('--checkpoint',
                        type=str,
                        default='bert-base-uncased',
                        required=True,
                        help='Needed to specify the number of layers and features')
    parser.add_argument('--num-labels',
                        default=2,
                        type=int)
    parser.add_argument('--max-seq-length',
                        default=219,
                        type=int,
                        help='Tokenization max length')
    parser.add_argument('--adapter-dim',
                        default = 0,
                        type = int,
                        help = 'Adapter Pooler dimension length')
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
                        help='input batch size for training (default: 16)')
    parser.add_argument('--lr',
                        type=float,
                        default=2e-5,
                        help='learning rate (default: 2e-5)')
    parser.add_argument('--num-workers',
                        type=int,
                        default=0,
                        help='H5 cannot be pickled')
    parser.add_mutually_exclusive_group(required=False)
    parser.add_argument('--error', dest='error', action='store_true')
    parser.add_argument('--no-error', dest='error', action='store_false')
    parser.set_defaults(feature=True)
    args = parser.parse_args()

    return args
#
