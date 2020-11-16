import argparse
import sys

sys.path.append('..')
import numpy as np
import pandas as pd

from keras import layers, optimizers, losses, metrics
from keras import Model


from keras.optimizers import RMSprop
from keras.optimizers import Adam

from keras.models import Sequential
from keras.models import load_model


from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization

from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

from keras.utils import to_categorical

from keras.losses import categorical_crossentropy

from keras import Input

from keras.backend import flatten

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split


from common.mnist_parser import *
from Autoencoder.encoder import *
from Autoencoder.decoder import *


# Build the model, consisti
def fully_connected(endoder, n_neurons):
	flat = Flatten()(endoder)
	dense = Dense(n_neurons, activation='relu')(flat)
	# dropout = Dropout(0.5, input_shape=(None, n_neurons))(dense);
	output = Dense(10, activation='softmax')(dense)

	return output


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

	train_X, rows, columns = parse_X(args.trainset)
	train_Y = parse_Y(args.trainlabels)
	test_X, _, _ = parse_X(args.testset)
	test_Y = parse_Y(args.testlabels)

	train_Y = to_categorical(train_Y)
	test_Y = to_categorical(test_Y)

	print("o sofo einai malakas ", train_Y.shape)

	# normalize all values between 0 and 1 
	train_X = train_X.astype('float32') / 255.
	test_X = test_X.astype('float32') / 255.
	
	# reshape the train and validation matrices into 28x28x1, due to an idiomorphy of the keras convolution.
	train_X = np.reshape(train_X, (len(train_X), rows, columns, 1))
	test_X = np.reshape(test_X, (len(test_X), rows, columns, 1))

	train_X ,valid_X, train_ground, valid_ground = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)


	loaded_encoder = load_model(args.model)
		
	while 1:
		found = False
		layers = loaded_encoder.layers
		for layer in layers:
			if 'dec' in layer.name:
				found = True
		if found:
			loaded_encoder._layers.pop()
		else:
			break

	input_img = Input(shape = (rows, columns, 1))
	
	encode = loaded_encoder(input_img)


	while 1:
			choice = int(input("Choose one of the following options to procced: \n \
						1 - Repeat the expirement with different hyperparameters\n \
						2 - Print the plots gathered from the expirement\n \
						3 - Save the model and exit\n"))
			if (choice == 1):
				epochs = int(input("Give the number of epochs\n"))
				batch_size = int(input("Give the batch size\n"))
				fc_n_neurons = int(input("Give the number of neurons in the fully connected layer\n"))

				full_model = Model(input_img, fully_connected(encode, fc_n_neurons))
				# 2 steps of training: 1st one, train only the fc layer
				for layer in full_model.layers:
					if (layer.name != 'dense'):
						layer.trainable = False

				full_model.compile(loss=categorical_crossentropy, optimizer=Adam(),metrics=['accuracy'])
				full_model.summary()

				classify_train = full_model.fit(train_X, train_ground, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_ground))

				# # 2nd step, train every layer
				for layer in full_model.layers:
					layer.trainable = True

				full_model.compile(loss=categorical_crossentropy, optimizer=Adam(),metrics=['accuracy'])
				
				classify_train = full_model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_X, test_Y))

				full_model.save( "../Models/classifier.h5", save_format='h5')

				break

			elif (choice == 2):
    				break;
			elif (choice == 3):
    				break;
			else:
				print("Choose one of the default values")
				continue


if __name__ == '__main__':
	main()