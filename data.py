# -*- coding: utf-8 -*-
# File:     data.py
# Author:   se7enXF
# Github:   se7enXF
# Date:     2019/6/20
# Note:

import os
from datetime import datetime
import tensorlayer as tl
import tensorflow as tf
import numpy as np


class DataSet:
	IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP', '.tif', '.tiff']
	IMG_MEAN = 255 / 2.0
	IMG_STD = 1

	def __init__(self, data_root, n_threads=32, image_size=512):
		self.__train_A = os.path.join(data_root, 'train_A')
		self.__train_B = os.path.join(data_root, 'train_B')
		self.__val_A = os.path.join(data_root, 'val_A')
		self.__val_B = os.path.join(data_root, 'val_B')
		self.__test_A = os.path.join(data_root, 'test_A')
		self.__test_B = os.path.join(data_root, 'test_B')
		self.__n_threads = n_threads
		self.img_size = image_size

	def __is_image_file(self, filename):
		return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)

	def __get_image_list(self, _dir):
		images = []
		assert os.path.isdir(_dir), f'{_dir} is not a valid directory'
		for root, _, f_names in sorted(os.walk(_dir)):
			for f in f_names:
				if self.__is_image_file(f):
					path = os.path.join(_dir, f)
					images.append(path)
		return images

	def __map_fn(self, img):
		img = tl.prepro.imresize(img, [self.img_size, self.img_size])
		img = img / self.IMG_MEAN - self.IMG_STD
		return img.astype(np.float32)

	def __get_images(self, f_a, f_b):
		files_a = self.__get_image_list(f_a)
		files_b = self.__get_image_list(f_b)
		img_a = tl.visualize.read_images(files_a, n_threads=self.__n_threads)
		img_b = tl.visualize.read_images(files_b, n_threads=self.__n_threads)
		img_a = tl.prepro.threading_data(data=img_a, fn=self.__map_fn)
		img_b = tl.prepro.threading_data(data=img_b, fn=self.__map_fn)
		return img_a, img_b

	def train(self):
		nom_a, nom_b = self.__get_images(self.__train_A, self.__train_B)
		(_n, _h, _w, _c) = nom_a.shape
		print(f'{datetime.now()}: Train data[{_n}, shape:({_h},{_w},{_c})]')
		return nom_a, nom_b

	def val(self):
		nom_a, nom_b = self.__get_images(self.__val_A, self.__val_B)
		(_n, _h, _w, _c) = nom_a.shape
		print(f'{datetime.now()}: Val data[{_n}, shape:({_h},{_w},{_c})]')
		return nom_a, nom_b

	def test(self):
		nom_a, nom_b = self.__get_images(self.__test_A, self.__test_B)
		(_n, _h, _w, _c) = nom_a.shape
		print(f'{datetime.now()}: Test data[{_n}, shape:({_h},{_w},{_c})]')
		return nom_a, nom_b


if __name__ == "__main__":
	d_root = 'D:/data/pix2pixHD'
	dataset = DataSet(d_root)
	test_x, test_y = dataset.test()
	print(len(test_x))


