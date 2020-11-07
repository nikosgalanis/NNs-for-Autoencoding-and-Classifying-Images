import argparse
import sys

sys.path.append('..')


from common.mnist_parser import *


parser = argparse.ArgumentParser(description='Create a NN for autoencoding a set of images')
parser.add_argument('dataset', metavar='-d', type=str, nargs='+', help='Path of the dataset')

args = parser.parse_args()

dataset_path = args.dataset[0]

dataset = parse_X(dataset_path)
# dataset.head()

print(dataset[:4])