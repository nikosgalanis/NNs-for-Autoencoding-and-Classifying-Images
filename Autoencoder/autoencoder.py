import argparse
import sys

sys.path.append('..')
# import keras
import numpy as np
import pandas as pd

from keras import layers, optimizers, losses, metrics
from keras.models import Sequential

from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.normalization import MaxPooling2D
from keras.layers import UpSampling2D


from common.mnist_parser import *


def encoder(input_image, conv_layers, conv_filter_size, n_conv_filters_per_layer):
	first_layer = Conv2D(n_conv_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same')(input_image)
	first_layer = BatchNormalization()(first_layer)
	first_layer = Conv2D(n_conv_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same')(first_layer)
	first_layer = BatchNormalization()(first_layer)

	pool = MaxPooling2D(pool_size=(2,2))(first_layer)

	enoding_layers = conv_layers

	current_filters_per_layer = 2 * n_conv_filters_per_layer

	for i in range (1, enoding_layers):
		conv_layer = Conv2D(current_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same')(pool)
		conv_layer = BatchNormalization()(conv_layer)
		conv_layer = Conv2D(current_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same')(conv_layer)
		conv_layer = BatchNormalization()(conv_layer)
		if (i < conv_layers - 1):
			pool = MaxPooling2D(pool_size=(2,2))(conv_layer)
		
		current_filters_per_layer *= 2


	return (conv_layer, enoding_layers, conv_filter_size, current_filters_per_layer)

def decoder(encoder_result):
	prev_conv_layer, decoding_layers, conv_filter_size, current_filters_per_layer = encoder_result

	current_filters_per_layer /= 2

	decoding_layers -= 1

	for _ in range (0, decoding_layers - 1):
		conv_layer = Conv2D(current_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same')(prev_conv_layer)
		conv_layer = BatchNormalization()(conv_layer)
		conv_layer = Conv2D(current_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same')(conv_layer)
		conv_layer = BatchNormalization()(conv_layer)


		prev_conv_layer = conv_layer

	upsampling = UpSampling2D((2,2))(conv_layer)

	# we want upsampling for the last layer
	last_conv_layer = Conv2D(current_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same')(upsampling)
	last_conv_layer = BatchNormalization()(last_conv_layer)
	last_conv_layer = Conv2D(current_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same')(last_conv_layer)
	last_conv_layer = BatchNormalization()(last_conv_layer)
	
	upsampling = UpSampling2D((2,2))(last_conv_layer)

	decoded = Conv2D(1, (conv_filter_size, conv_filter_size), activation='sigmoid', padding='same')(upsampling)

	return decoded

def main():
	# Create an arguments' parser
	parser = argparse.ArgumentParser(description='Create a NN for autoencoding a set of images')

	# only needed argument: -d for the dataset
	parser.add_argument('-d', '--dataset', action='store', default=None,  metavar='', help='Relative path to the dataset')

	args = parser.parse_args()

	dataset = parse_X(args.dataset)

	# get the hyperparameters from the user
	conv_layers = input("Give the number of convolution layers")
	conv_filter_size = input("Give the size of each convolution filter")
	n_conv_filters_per_layer = input("Give the number of convolution filters per layer")
	epochs = input("Give the number of epochs")
	batch_size = input("Give the batch size")

	model = Sequential()

	# autoencoder = model()


# Main function of the autoencoder
if __name__ == "__main__":
	main()
