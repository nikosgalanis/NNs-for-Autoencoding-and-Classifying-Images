import argparse
import sys

sys.path.append('..')
import numpy as np
import pandas as pd

from keras import layers, optimizers, losses, metrics
from keras.models import Sequential
from keras import Model
from keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split

from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D

import tensorflow as tf

from common.mnist_parser import *

# encoding function for the autoencoder
"""
FUNCTION USAGE: The user gives the desired hyperparameters as following:
 - input_image: the shape of the images that the model is going to predict later
 - conv_layers: the number of convolution layers that the encoder will apply to the image
 - conv_filter_size: The size of the filter during each one of the convolution layers
 - n_conv_filters_per_layer: The number of filters in the first layer, each time multiplied by 2

 The encoder applies convolution, and batch normalization as many times as the 
 user has requested, and applies pooling after the 2 first layers, and returns 
 a tuple containing all the usefull info (result + hyperparameters).
"""
def encoder(input_image, conv_layers, conv_filter_size, n_conv_filters_per_layer):

	# first layer: differs because it takes as input the shape of the image
	first_layer = Conv2D(n_conv_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same')(input_image)
	first_layer = BatchNormalization()(first_layer)
	first_layer = Conv2D(n_conv_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same')(first_layer)
	first_layer = BatchNormalization()(first_layer)

	# pooling after first layer
	pool = MaxPooling2D(pool_size=(2,2))(first_layer)

	# the encoding layers are the number of layers given to us as an argument (TODO: possibly change)
	encoding_layers = conv_layers

	# each time, we will be nultiplying the filters per layer by 2
	current_filters_per_layer = 2 * n_conv_filters_per_layer

	# for the remaining encoding layers
	for i in range (1, encoding_layers):
		# the first 2 take an input from the pooling
		if (i == 1 or i == 2):
			conv_layer = Conv2D(current_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same')(pool)
		# the others from the previous convolution layer
		else:
			conv_layer = Conv2D(current_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same')(conv_layer)

		# after that, wy apply batch normalization, convolution and the again batch normalization
		conv_layer = BatchNormalization()(conv_layer)
		conv_layer = Conv2D(current_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same')(conv_layer)
		conv_layer = BatchNormalization()(conv_layer)
		print(i, current_filters_per_layer)
		# on the 1st layer in the loop(aka the 2nd), we want pooling
		if (i == 1):
			pool = MaxPooling2D(pool_size=(2,2))(conv_layer)
		# each time (except the last one, we multiply the filters per layer by 2)
		if (i < encoding_layers - 1):
			current_filters_per_layer *= 2

	# return a tuple containing all the usefull info that we gathered from the encoder
	return (conv_layer, encoding_layers, conv_filter_size, current_filters_per_layer)


# decoding function for the autoencoder
"""
FUNCTION USAGE: The function recieves an encoder result, as a tuple, with the following 
parameters included:
 - prev_conv_layer: The shape and the characteristics of the matrix produced by the last encoding step
 - decoding_layers: the number of convolution layers that the decoder will apply to the image
 - conv_filter_size: The size of the filter during each one of the convolution layers
 - n_conv_filters_per_layer: The number of filters that will be used in the 1st step of the decoder,
	 each time divided by 2

 The decoder applies convolution, and batch normalization as many times as the 
 user has requested, and applies upasmpling after the last 2 layers, and returns 
 the shape of our array after the decoding proccedure, along with all its data
"""
def decoder(encoder_result):
	print("Decoder")
	# gather the info given by the encoder tuple
	prev_conv_layer, decoding_layers, conv_filter_size, current_filters_per_layer = encoder_result

	# divide the filters by 2
	current_filters_per_layer /= 2

	# the decoding layers are one less than the encoding ones
	decoding_layers -= 1

	# for the first n-1 layers of the decoder
	for i in range (0, decoding_layers - 1):
		# apply convolution and batch normalization 2 times
		conv_layer = Conv2D(current_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same')(prev_conv_layer)
		conv_layer = BatchNormalization()(conv_layer)
		conv_layer = Conv2D(current_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same')(conv_layer)
		conv_layer = BatchNormalization()(conv_layer)
		print(i, current_filters_per_layer)
		
		# again, devide the filters per layer by 2
		current_filters_per_layer /= 2

		# to satisfy the next loop, the current layer becomes the previous one
		prev_conv_layer = conv_layer

	# after the completion of the loop, apply an upsampling technique
	upsampling = UpSampling2D((2,2))(prev_conv_layer)

	print(current_filters_per_layer)
 
	# the last layer takes its input from the upsampling that we've performed
	last_conv_layer = Conv2D(current_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same')(upsampling)
	last_conv_layer = BatchNormalization()(last_conv_layer)
	last_conv_layer = Conv2D(current_filters_per_layer, (conv_filter_size, conv_filter_size), activation='relu', padding='same')(last_conv_layer)
	last_conv_layer = BatchNormalization()(last_conv_layer)
	
	# apply one last time the upsampling technique
	upsampling = UpSampling2D((2,2))(last_conv_layer)

	# the decoded array is produced by applying 2d convolution one last time, this one with a sigmoid activation function
	decoded = Conv2D(1, (conv_filter_size, conv_filter_size), activation='sigmoid', padding='same')(upsampling)

	return decoded


# Main function of the autoencoder
"""
FUNCTION USAGE: The main function is given a path of a file containing the 
MNINT dataset, and tries to apply autoencoding for those images in the dataset.
This is an expiramental approach with respect to the hyperparameters of the
problem, thus the program will run many loops, train many models, and plot 
the loss function results, until the user is satisfied by the output, and selects
to save that model for later usage.
"""
def main():
	# parse the file in order 
	dataset = parse_X("train-images-idx3-ubyte")

	# get the hyperparameters from the user
	conv_layers = 4 # input("Give the number of convolution layers")
	conv_filter_size = 3 # input("Give the size of each convolution filter")
	n_conv_filters_per_layer = 32 #input("Give the number of convolution filters per layer")
	epochs = 5 #input("Give the number of epochs")
	batch_size = 256 #input("Give the batch size")
	
	# define the shape of the images that we are going to use
	from keras import Input
	input_img = Input(shape=(28, 28, 1))

	# from keras.datasets import mnist
	# (train_X, _), (valid_X, _) = mnist.load_data()

	# the autoencoder is a keras model class, consisted of an encoder and a decoder
	autoencoder = Model(input_img, decoder(encoder(input_img, conv_layers, conv_filter_size, n_conv_filters_per_layer)))
	# compile the model
	autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())
	print("KOMPLE")
 
	# split the dataset in order to check the model's behaviour
	train_X, valid_X, train_ground, valid_ground = train_test_split(dataset, dataset, test_size=0.95, random_state=42)
	
	#	normalize all values between 0 and 1 
	train_X = train_X.astype('float32') / 255.
	valid_X = valid_X.astype('float32') / 255.
	
	# reshape the train and validation matrices into 28x28x1, due to an idiomorphy of the keras convolution.
	train_X = np.reshape(train_X, (len(train_X), 28, 28, 1))
	valid_X = np.reshape(valid_X, (len(valid_X), 28, 28, 1))

	# fit the problem in order to check its behaviour
	auto_train = autoencoder.fit(train_X, train_X, batch_size=batch_size, epochs=epochs, validation_data=(valid_X, valid_X))


# Main function of the autoencoder
if __name__ == "__main__":
	main()
