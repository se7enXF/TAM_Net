# -*- coding: utf-8 -*-
# File:     train.py
# Author:   se7enXF
# Github:   se7enXF
# Date:     2019/6/18

import os
import time
import argparse
from model import TAMNet
from data import DataSet
import tensorlayer as tl
import tensorflow as tf
import numpy as np
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='work_3')
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--data_dir', type=str, default='D:/pix2pixHD/datasets/image2road')
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--eval_train', type=bool, default=False)
parser.add_argument('--save_model_freq', type=int, default=50)
flags = parser.parse_args()

# make dirs
tb_dir = os.path.join(flags.log_dir, flags.name, 'tensorboard')
weight_dir = os.path.join(flags.log_dir, flags.name, 'checkpoint')
tl.files.exists_or_mkdir(tb_dir)
tl.files.exists_or_mkdir(weight_dir)

# data and tensorboard
print(f'{datetime.now()}: Reading dataset ...')
dataset = DataSet(flags.data_dir, n_threads=32, image_size=flags.image_size)
x_train, y_train = dataset.train()
train_writer = tf.summary.create_file_writer(tb_dir + '/train')
if flags.eval_train:
	x_val, y_val = dataset.val()
	val_writer = tf.summary.create_file_writer(tb_dir + '/validation')
else:
	x_val, y_val = None, None
	val_writer = None
n_data = len(x_train)

# network
print(f"{datetime.now()}: Creating network ...")
tam_net = TAMNet([None, flags.image_size, flags.image_size, 3])
network = tam_net.model
criterion = tam_net.criterion
train_op = tf.optimizers.Adam(flags.lr)
print(network)

# train the net work
print(f"{datetime.now()}: Start training the network ...")
print(f"{datetime.now()}: Use `tensorboard --logdir={tb_dir}` to start tensorboard")

start_date = datetime.now()
for e in range(flags.epoch):
	start_time = time.time()

	# train
	g_loss, train_iter = 0, 0
	for x_batch, y_batch in tl.iterate.minibatches(x_train, y_train, flags.batch_size, shuffle=True):
		with tf.GradientTape() as tape:
			outs = network(x_batch, is_train=True)
			image_loss = criterion(outs, y_batch)
			g_loss += image_loss
			train_iter += 1
		loss = g_loss
		grad = tape.gradient(loss, network.trainable_weights)
		train_op.apply_gradients(zip(grad, network.trainable_weights))
	cost_time = round(time.time() - start_time, 4)
	g_loss = g_loss/train_iter
	print(f"{datetime.now()}: Epoch [{e + 1}/{flags.epoch}] took {cost_time} seconds. Image Loss: {g_loss}")

	# write and print train summary
	with train_writer.as_default():
		tf.summary.scalar('Train Loss', g_loss, step=e + 1)
		tf.summary.image('Train Sample', [x_batch[0], y_batch[0], outs[0]], step=e + 1)

	# evaluation
	if flags.eval_train:
		val_loss, n_iter = 0, 0
		for x_batch, y_batch in tl.iterate.minibatches(x_val, y_val, flags.batch_size, shuffle=False):
			outs = network(x_batch, is_train=False)
			loss = criterion(outs, y_batch)
			val_loss += loss
			n_iter += 1
		rmse = val_loss / n_iter
		with val_writer.as_default():
			tf.summary.scalar('Val Loss', rmse, step=e+1)
			tf.summary.image('Val Sample', [x_batch[0], y_batch[0], outs[0]], step=e + 1)
		print(f"{datetime.now()}: Valuation Loss: {rmse}")

	# save model
	if (e+1) % flags.save_model_freq == 0 or e+1 == flags.epoch:
		save_dir = os.path.join(weight_dir, 'epoch_{}.h5'.format(e+1))
		network.save_weights(save_dir)
		print(f"{datetime.now()}: Model file saved at:{save_dir}")

cost_time = datetime.now() - start_date
print(f"{datetime.now()}: Training done. Total time {cost_time}")
