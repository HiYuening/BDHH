from nbr.trainer import NBRTrainer
import argparse
import pickle
import time
from util import Data, split_validation
from model import *
import os
from data_loader import dataLoader
from trainer import training


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

same_seeds(1234)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Dunnhumby', help='')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size') #8
parser.add_argument('--emb_size', type=int, default=128, help='embedding size')
parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads')
parser.add_argument('--l2', type=float, default=1e-3, help='l2 penalty')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--layer', type=float, default=1, help='the number of layer used, 2 for amazon, 3 for others')
parser.add_argument('--beta', type=float, default=0.01, help='ssl task maginitude')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')
parser.add_argument('--opt', type=str, default='Adam')
parser.add_argument('--evalEpoch', type=int, default=1)
parser.add_argument('--isTrain', type=int, default=0)

config = parser.parse_args()
print(config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Select the first 10 baskets for each user in the Dunnhumby dataset，K=10
ID starts from 1 in dataset
len(session):  2463
len(item):  12150 (removed duplicates, contain historical and targeting items)
len(basket):  24198
len(category):  235 (removed duplicates, contain historical and targeting category)
len(price):  10 (removed duplicates, contain historical and targeting category)
'''

def main():
    dataset = dataLoader(config)

    print('start training')
    training(dataset, config, device,n_node=12150,n_price=10,n_category=235) # change these values if you change another dataset


if __name__ == '__main__':
    main()
