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
from keras import Input

from matplotlib import pyplot as plt

import tensorflow as tf

from common.mnist_parser import *
from common.utils import *

from Autoencoder.encoder import *
from Autoencoder.decoder import *

try:
	import hiplot as hip
except:
	# pip install hiplot
	install('hiplot')
	import hiplot as hip


import webbrowser

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
	# create a parser in order to obtain the arguments
	parser = argparse.ArgumentParser(description='Create a NN for autoencoding a set of images')
	# the oonly argument that we want is -d
	parser.add_argument('-d', '--dataset', action='store', default=None,  metavar='', help='Relative path to the dataset')
	# parse the arguments
	args = parser.parse_args()

	# parse the MNIST dataset and obtain its rows and columns
	dataset, rows, columns = parse_X(args.dataset)

	# define the shape of the images that we are going to use
	input_img = Input(shape=(rows, columns, 1))

	# # split the dataset in order to check the model's behaviour
	train_X, valid_X, train_ground, valid_ground = train_test_split(dataset, dataset, test_size=0.2, random_state=13)
	
	# normalize all values between 0 and 1 
	train_X = train_X.astype('float32') / 255.
	valid_X = valid_X.astype('float32') / 255.
	
	# reshape the train and validation matrices into 28x28x1, due to an idiomorphy of the keras convolution.
	train_X = np.reshape(train_X, (len(train_X), rows, columns, 1))
	valid_X = np.reshape(valid_X, (len(valid_X), rows, columns, 1))

	# list to keep all of our models
	models_list = []
	plots = 0
	#boolean variable to exit the program
	offside = False
	# run until the user decides to break
	while (offside == False):
		# get the hyperparameters from the user
		conv_layers = int(input("Give the number of convolution layers\n"))
		conv_filter_size = int(input("Give the size of each convolution filter\n"))
		n_conv_filters_per_layer = int(input("Give the number of convolution filters per layer\n"))
		epochs = int(input("Give the number of epochs\n"))
		batch_size = int(input("Give the batch size\n"))

		print ("---TRYING TO RUN THE AUTOENCODER WITH THE FOLLOWING PARAMETERS: \nconv_layers ", conv_layers, \
			"   conv_filter_size: ", conv_filter_size, "   n_conv_filters_per_layer: ", n_conv_filters_per_layer, \
				"   epochs: ", epochs, "   batch_size: ", batch_size)
		
		try:
			# the autoencoder is a keras model class, consisted of an encoder and a decoder
			autoencoder = Model(input_img, decoder(encoder(input_img, conv_layers, conv_filter_size, n_conv_filters_per_layer)))

			# compile the model
			# theory has shown that the best optimizer for the mnist dataset is the following
			autoencoder.compile(loss='categorical_crossentropy', optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0))

			# visualize the layers that we've created using summary()
			autoencoder.summary()

			# fit the problem in order to check its behaviour
			auto_train = autoencoder.fit(train_X, train_X, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_X))
			# create a tuple of the model plus some info in order to save it to the models' list
			# model_plus_info = (autoencoder, conv_layers, conv_filter_size, n_conv_filters_per_layer, epochs, batch_size)
			# create a dictionary of the model plus some info in order to save it to the models' list
			model_plus_info = {'full_model': autoencoder, 'epochs': epochs, 'batch_size': batch_size,\
								'convolution layers': conv_layers, 'convolution filter size ': conv_filter_size,\
								'number of conv. filters per layer': n_conv_filters_per_layer}
			models_list.append(model_plus_info)

		except:
			# an error has occured, but we dont want to exit the program
			print("Chosen hyperparameters caused an error. Please select others")
			
		# ask the user for new input
		while 1:
			choice = int(input("Experiment completed! Choose one of the following options to procced: \n \
							1 - Repeat the expirement with different hyperparameters\n \
							2 - Print the plots gathered from the expirement\n \
							3 - Save the model and exit\n \
							4 - Load a pre-trained model\n"))
			if (choice == 1):
				break
			elif (choice == 2):
				if (len(models_list) > 0):
					print("\n------------------TRAINED MODEL METRICS----------------------")
			
					# Loss plot
					plt.figure()
					plt.xlabel('Epochs')
					plt.ylabel('MSE loss')
					title = print(" Training/validation loss iconv_layers: " + str(conv_layers) + \
						" conv_filter_size: " + str(conv_filter_size) + \
						" n_conv_filters_per_layer: " + str(n_conv_filters_per_layer) + \
						" epochs: " + str(epochs) + " batch_size: " + str(batch_size))
					plt.title(title)
					plt.plot(auto_train.history['loss'], label = 'Training loss')
					plt.plot(auto_train.history['val_loss'], label = 'Validation loss')
					plt.legend(loc="upper right")
					plt.legend()
					# path to save the plot image
					plot_path = "loss_plot_" + str(conv_layers) + "_" +  str(conv_filter_size) + "_" +  str(n_conv_filters_per_layer) +  ".png"
					# save the image
					plt.savefig(plot_path)
					plt.show()

					# Accuracy plot
					plt.figure()
					plt.xlabel('Epochs')
					plt.ylabel('MSE loss')
					title = print(" Training/validation accuracy iconv_layers: " + str(conv_layers) + \
										" conv_filter_size: " + str(conv_filter_size) + \
										" n_conv_filters_per_layer: " + str(n_conv_filters_per_layer) + \
										" epochs: " + str(epochs) + " batch_size: " + str(batch_size))
					plt.title(title)
					plt.plot(auto_train.history['accuracy'], label = 'Training accuracy')
					plt.plot(auto_train.history['val_accuracy'], label = 'Validation accuracy')
					plt.legend(loc="lower right")
					plt.legend()
					# path to save the plot image
					plot_path = "accuracy_plot_" + str(conv_layers) + "_" +  str(conv_filter_size) + "_" +  str(n_conv_filters_per_layer) +  ".png"
					# save the image
					plt.savefig(plot_path)
					plt.show()
					
					if (len(models_list) > 1):
						print("\n--------------- ALL TRAINED MODElS METRICS-----------------\n")

						"""
						Using hiPlot to create a high dimensional plot of our models' hyperparameters
						"""
						plotting_dict_list = []
						for m in models_list:
							m["loss"] = round(m['full_model'].history.history['loss'][-1], 4)
							m["accuracy"] = round(m['full_model'].history.history['accuracy'][-1], 4)
							plotting_dict_list.append(dict(list(m.items())[1:]))


						html_str = hip.Experiment.from_iterable(plotting_dict_list).to_html(force_full_width=True)
						
						# open a file to save the html containing the plot 
						f_name = "hiplot_result_" + str(plots) + '.html'
						# increase the plot variable
						plots += 1
						f = open(f_name, "w")
						f.write(html_str)
						f.close()
						# pop-up in google chrome
						webbrowser.get('/usr/bin/google-chrome %s').open(f_name)
					else :
						print(len(models_list))
						print("There is only one model, train an other one too to compare...\n")
					
					continue
			elif (choice == 3):
    			# demand a datapath to save the model
				path = input("Give the path and the name of the file to save the model\n")
				autoencoder.save(path + ".h5", save_format='h5')
				# break the loop: model training is finished
				offside = True
				break
			elif (choice == 4):
				# Get the model info
				path = input("Give the path and the name of the model you want to load")	
				conv_layers = int(input("Give the number of convolutional layers that were used to train the model"))
				conv_filter_size = int(input("Give the conv_filter_size that was used to train the model"))
				n_conv_filters_per_layer = int(input("Give the number of convolution filters per layer that were used to train the model"))
				epochs = int(input("Give the number of epochs that were used to train the model"))
				batch_size = int(input("Give the batch size that was used to train the model"))
				# load the pre-trained model
				autoencoder = load_model(path)
				# collect the info in the tuple
				model_plus_info = (autoencoder, conv_layers, conv_filter_size, n_conv_filters_per_layer, epochs, batch_size)
				# append the model in the models' list
				models_list.append(model_plus_info)
			else:
				print("Choose one of the default values")
				continue


"""

[a, b, c] -> [10,20,30]

[d, e, f] -> [10,20,40]

plot(x:epochs, y: error)

"""

# Main function of the autoencoder
if __name__ == "__main__":
	main()
