import os, torch
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="BERTVision for PyTorch")
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')

    parser.add_argument('--batch-size', type=int, default=16, help='input batch size for training (default: 16)')

    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate (default: 1e-5)')

    parser.add_argument('--num-workers', type=int, default=0, help='Number of CPU cores (default: 0) H5PY cannot pickle')

    parser.add_argument('--l2', type=float, default=0.01, help='l2 regularization weight (default: 0.01)')

    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    parser.add_argument('--patience', type=int, default=3, help="Number of epochs\
                        to use as the basis for early stopping (default: 3)")

    return parser
