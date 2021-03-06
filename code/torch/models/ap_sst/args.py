import os
import sys
sys.path.append("C:\\BERTVision")
import torch
import models.args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    # retreive the general models.args and attach them here
    parser = models.args.get_args()
    # set some AP_SST specific args
    parser.add_argument('--model', default=None, type=str, required=True)
    parser.add_argument('--device', default=device)
    parser.add_argument('--dataset', type=str, default='SSTH5', choices=['SSTH5'])
    parser.add_argument('--model-name', default='ap_sst', type=str)
    parser.add_argument('--save-path', type=str, default=os.path.join('model_checkpoints'))
    # runs sst-5 with --is-multilabel
    parser.add_argument('--is-multilabel', default=False, action='store_true')
    parser.add_argument('--is-binary', dest='is-multilabel', action='store_true')
    
    parser.add_argument('--warmup-proportion',
                        default=0.1,
                        type=float,
                        help='Proportion of training to perform linear learning rate warmup for')
    args = parser.parse_args()
    return args
#
