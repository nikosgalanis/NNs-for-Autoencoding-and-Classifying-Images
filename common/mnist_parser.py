import os
import struct

import numpy as np

def read_big_endian(byte_sequence):
	return int.from_bytes(byte_sequence, "big")

def parse_X(path):
	with open(path, "rb") as file:
		magic_number = read_big_endian(file.read(4))
		if magic_number < 0:
			raise Exception("Wrong magic number!")


		n_of_images = read_big_endian(file.read(4))
		rows = read_big_endian(file.read(4))
		columns = read_big_endian(file.read(4))
		
		result = np.zeros((n_of_images, rows, columns))
		curr_row = curr_col = curr_image = 0


		while (curr_image < n_of_images):
			number = read_big_endian(file.read(1))
			result[curr_image, curr_row, curr_col] = number

			if (curr_row == rows - 1 and curr_col == columns - 1):
				curr_image += 1
				curr_row = 0
				curr_col = 0
			elif (curr_col == columns - 1):
				curr_row += 1
				curr_col = 0
			else:
				curr_col += 1
	
	return result


def parse_Y(path):
	with open(path, "rb") as file:
		magic_number = read_big_endian(file.read(4))
		if magic_number < 0:
			raise Exception("Wrong magic number!")

		n_of_labels = read_big_endian(file.read(4))

		result = np.zeros((n_of_labels))
		current_label = 0

		while current_label < n_of_labels:
			number = read_big_endian(file.read(1))
			result[current_label] = number
			current_label += 1

	return result

