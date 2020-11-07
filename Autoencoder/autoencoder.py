import argparse
import sys

sys.path.append('..')


from common.mnist_parser import *


parser = argparse.ArgumentParser(description='Create a NN for autoencoding a set of images')

parser.add_argument('-d', '--dataset', action='store', default=None,  metavar='', help='Relative path to the dataset')

args = parser.parse_args()

dataset = parse_X(args.dataset)

print(dataset[:4])

conv_layers = input("Give the number of convolution layers")
conv_filter_size = input("Give the size of each convolution filter")
n_conv_filters_per_layer = input("Give the number of convolution filters per layer")
epochs = input("Give the number of epochs")
batch_size = input("Give the batch size")
