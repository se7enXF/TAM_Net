# -*- coding: utf-8 -*-
# File:     train.py
# Author:   se7enXF
# Github:   se7enXF
# Date:     2019/6/18
# Note:

from model import TAMNet
from data import DataSet
import argparse
import numpy as np
import tensorlayer as tl

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='D:/pix2pixHD/datasets/image2road')
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--model_dir', type=str, default='logs/work_1/checkpoint/epoch_500.h5')
flags = parser.parse_args()

# data
dataset = DataSet(flags.data_dir, n_threads=32, image_size=flags.image_size)
x_test, y_test = dataset.test()

# network
tam_net = TAMNet([None, flags.image_size, flags.image_size, 3])
network = tam_net.model
network.load_weights(flags.model_dir)

# train the net work

out = tl.utils.predict(network=network, X=x_test, batch_size=1)
for i in range(len(x_test)):
	img = np.concatenate([x_test[i], y_test[i], out[i]], axis=1)
	tl.vis.frame(img, second=2, saveable=False)
	# tl.vis.save_image(out[0], 'test.jpg')
