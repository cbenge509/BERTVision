import os
import sys
sys.path.append("C:\\BERTVision")
import torch
import models.args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    # retreive the general models.args and attach them here
    parser = models.args.get_args()
    # set some AP specific args
    parser.add_argument('--model', default=None, type=str, required=True)
    parser.add_argument('--device', default=device)
    parser.add_argument('--dataset', type=str, default='SQuADH5', choices=['SQuADH5'])
    parser.add_argument('--model-name', default='AP', type=str)
    parser.add_argument('--save-path', type=str, default=os.path.join('model_checkpoints'))
    parser.add_argument('--warmup-proportion',
                        default=0.1,
                        type=float,
                        help='Proportion of training to perform linear learning rate warmup for')
    args = parser.parse_args()
    return args
#
