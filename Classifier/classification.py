import argparse
import sys

sys.path.append('..')
import numpy as np
import pandas as pd

from keras import layers, optimizers, losses, metrics
from keras.models import Sequential
from keras import Model
from keras.optimizers import RMSprop
from keras.models import load_model

from sklearn.model_selection import train_test_split

from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import pop

from keras import Input

from matplotlib import pyplot as plt

import tensorflow as tf

from common.mnist_parser import *

from Autoencoder.encoder import *
from Autoencoder.decoder import *


def main():
	# create a parser in order to obtain the arguments
	parser = argparse.ArgumentParser(description='Create a an image classifier NN using a pre-trained encoder')
	# the oonly argument that we want is -d
	parser.add_argument('-d', '--trainset', action='store', default=None,  metavar='', help='Relative path to the training set')
	parser.add_argument('-dl', '--trainlabels', action='store', default=None,  metavar='', help='Relative path to the training labels')
	parser.add_argument('-t', '--testset', action='store', default=None,  metavar='', help='Relative path to the test set')
	parser.add_argument('-tl', '--testlabels', action='store', default=None,  metavar='', help='Relative path to the test labels')
	parser.add_argument('-model', '--model', action='store', default=None,  metavar='', help='Relative path to the h5 file of the trained encoder model')

	# parse the arguments
	args = parser.parse_args()

	train_X = parse_X(args.trainset)
	train_Y = parse_Y(args.trainlabels)
	test_X = parse_X(args.testset)
	test_Y = parse_Y(args.testlabels)

	loaded_encoder = load_model(args.model)
	loaded_encoder.summary()



if __name__ == '__main__':
	main()