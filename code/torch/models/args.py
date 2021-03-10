import os, torch
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="BERTVision for PyTorch")
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')

    parser.add_argument('--l2', type=float, default=0.01, help='l2 regularization weight (default: 0.01)')

    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    parser.add_argument('--patience', type=int, default=3, help="Number of epochs\
                        to use as the basis for early stopping (default: 3)")
    return parser
