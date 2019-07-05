# -*- coding: utf-8 -*-
# File:     model.py
# Author:   se7enXF
# Github:   se7enXF


import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Input, Conv2d,  DeConv2d, Concat, BatchNorm, MaxPool2d, Elementwise
from tensorlayer.models import Model


class TAMNet(object):
	def __init__(self, in_shape=None):
		self.in_shape = in_shape
		self.f_size = 64
		self.model = self.tam_net()

	def tam_net(self):
		inputs = Input(self.in_shape, name='inputs')

		e_in = inputs
		for i in range(0, 5):
			e_out = Conv2d(self.f_size * (2**i), (3, 3), (2, 2), act=tf.nn.relu, name=f'e{i+1}_con')(e_in)
			e_in = self.residual_block(i, e=True)(e_out)
			self.__setattr__(f'e{i+1}', e_in)

		d_in = e_in
		for i in range(4, 0, -1):
			d_out = DeConv2d(self.f_size * (2**(i-1)), (3, 3), (2, 2), name=f'd{i}_con')(d_in)
			encoder = self.__getattribute__(f'e{i}')
			d_out = Concat(concat_dim=3, name=f'concat{i}')([encoder, d_out])
			d_out = Conv2d(self.f_size * (2**(i-1)), (1, 1), (1, 1), name=f'fusion{i}')(d_out)
			d_in = self.residual_block(i-1, e=False)(d_out)
			self.__setattr__(f'd{i + 1}', d_in)

		outs = DeConv2d(3, (3, 3), (2, 2), name='d_con_out')(d_in)
		outs = Conv2d(3, (1, 1), (1, 1), act=tf.nn.sigmoid, name='outs')(outs)
		return Model(inputs=inputs, outputs=outs, name="TAM_Net")

	def residual_block(self, n_k=1, e=True):
		k_size = self.f_size*(2**n_k)
		ni = Input([None, None, None, k_size])
		nn = Conv2d(k_size, (3, 3), (1, 1))(ni)
		nn = BatchNorm(act=tf.nn.relu)(nn)
		nn = Conv2d(k_size, (3, 3), (1, 1))(nn)
		nn = BatchNorm()(nn)
		nn = Elementwise(tf.add)([ni, nn])
		return Model(inputs=ni, outputs=nn, name=f'{"e" if e else "d"}{n_k+1}_res').as_layer()

	def criterion(self, x, y):
		mse = tl.cost.mean_squared_error(x, y, is_mean=True)
		rmse = tf.sqrt(mse, name='RMSE')
		return rmse


class UNet(object):
	def __init__(self, in_shape=None, is_bn=False, is_train_bn=False):
		self.in_shape = in_shape
		self.train_bn = False
		self.model = self.model_select(is_bn, is_train_bn)

	def model_select(self, is_bn, is_train_bn):
		if is_bn:
			self.train_bn = is_train_bn
			return self.batch_normal_u_net()
		else:
			return self.u_net()

	def u_net(self):
		inputs = Input(self.in_shape, name='inputs')

		conv1 = Conv2d(64, (3, 3), act=tf.nn.relu, name='conv1_1')(inputs)
		conv1 = Conv2d(64, (3, 3), act=tf.nn.relu, name='conv1_2')(conv1)
		pool1 = MaxPool2d((2, 2), name='pool1')(conv1)

		conv2 = Conv2d(128, (3, 3), act=tf.nn.relu, name='conv2_1')(pool1)
		conv2 = Conv2d(128, (3, 3), act=tf.nn.relu, name='conv2_2')(conv2)
		pool2 = MaxPool2d((2, 2), name='pool2')(conv2)

		conv3 = Conv2d(256, (3, 3), act=tf.nn.relu, name='conv3_1')(pool2)
		conv3 = Conv2d(256, (3, 3), act=tf.nn.relu, name='conv3_2')(conv3)
		pool3 = MaxPool2d((2, 2), name='pool3')(conv3)

		conv4 = Conv2d(512, (3, 3), act=tf.nn.relu, name='conv4_1')(pool3)
		conv4 = Conv2d(512, (3, 3), act=tf.nn.relu, name='conv4_2')(conv4)
		pool4 = MaxPool2d((2, 2), name='pool4')(conv4)

		conv5 = Conv2d(1024, (3, 3), act=tf.nn.relu, name='conv5_1')(pool4)
		conv5 = Conv2d(1024, (3, 3), act=tf.nn.relu, name='conv5_2')(conv5)

		up4 = DeConv2d(512, (3, 3), (2, 2), name='deconv4')(conv5)
		up4 = Concat(3, name='concat4')([up4, conv4])
		conv4 = Conv2d(512, (3, 3), act=tf.nn.relu, name='uconv4_1')(up4)
		conv4 = Conv2d(512, (3, 3), act=tf.nn.relu, name='uconv4_2')(conv4)

		up3 = DeConv2d(256, (3, 3), (2, 2), name='deconv3')(conv4)
		up3 = Concat(3, name='concat3')([up3, conv3])
		conv3 = Conv2d(256, (3, 3), act=tf.nn.relu, name='uconv3_1')(up3)
		conv3 = Conv2d(256, (3, 3), act=tf.nn.relu, name='uconv3_2')(conv3)

		up2 = DeConv2d(128, (3, 3), (2, 2), name='deconv2')(conv3)
		up2 = Concat(3, name='concat2')([up2, conv2])
		conv2 = Conv2d(128, (3, 3), act=tf.nn.relu, name='uconv2_1')(up2)
		conv2 = Conv2d(128, (3, 3), act=tf.nn.relu, name='uconv2_2')(conv2)

		up1 = DeConv2d(64, (3, 3), (2, 2), name='deconv1')(conv2)
		up1 = Concat(3, name='concat1')([up1, conv1])
		conv1 = Conv2d(64, (3, 3), act=tf.nn.relu, name='uconv1_1')(up1)
		conv1 = Conv2d(64, (3, 3), act=tf.nn.relu, name='uconv1_2')(conv1)

		outs = Conv2d(3, (1, 1), act=tf.nn.sigmoid, name='uconv1')(conv1)
		return Model(inputs=inputs, outputs=outs, name="U_Net")

	def batch_normal_u_net(self):
		# built encoder
		inputs = Input(self.in_shape, name='inputs')
		conv1 = Conv2d(64, (4, 4), (2, 2), name='conv1')(inputs)

		conv2 = Conv2d(128, (4, 4), (2, 2), name='conv2')(conv1)
		conv2 = BatchNorm(act=tl.act.lrelu, is_train=self.train_bn, name='bn2')(conv2)

		conv3 = Conv2d(256, (4, 4), (2, 2), name='conv3')(conv2)
		conv3 = BatchNorm(act=tl.act.lrelu, is_train=self.train_bn, name='bn3')(conv3)

		conv4 = Conv2d(512, (4, 4), (2, 2), name='conv4')(conv3)
		conv4 = BatchNorm(act=tl.act.lrelu, is_train=self.train_bn, name='bn4')(conv4)

		conv5 = Conv2d(512, (4, 4), (2, 2), name='conv5')(conv4)
		conv5 = BatchNorm(act=tl.act.lrelu, is_train=self.train_bn, name='bn5')(conv5)

		conv6 = Conv2d(512, (4, 4), (2, 2), name='conv6')(conv5)
		conv6 = BatchNorm(act=tl.act.lrelu, is_train=self.train_bn, name='bn6')(conv6)

		conv7 = Conv2d(512, (4, 4), (2, 2), name='conv7')(conv6)
		conv7 = BatchNorm(act=tl.act.lrelu, is_train=self.train_bn, name='bn7')(conv7)

		conv8 = Conv2d(512, (4, 4), (2, 2), act=tl.act.lrelu, name='conv8')(conv7)

		# built decoder
		up7 = DeConv2d(512, (4, 4), name='deconv7')(conv8)
		up7 = BatchNorm(act=tf.nn.relu, is_train=self.train_bn, name='dbn7')(up7)

		up6 = Concat(concat_dim=3, name='concat6')([up7, conv7])
		up6 = DeConv2d(1024, (4, 4), name='deconv6')(up6)
		up6 = BatchNorm(act=tf.nn.relu, is_train=self.train_bn, name='dbn6')(up6)

		up5 = Concat(concat_dim=3, name='concat5')([up6, conv6])
		up5 = DeConv2d(1024, (4, 4), name='deconv5')(up5)
		up5 = BatchNorm(act=tf.nn.relu, is_train=self.train_bn, name='dbn5')(up5)

		up4 = Concat(concat_dim=3, name='concat4')([up5, conv5])
		up4 = DeConv2d(1024, (4, 4), name='deconv4')(up4)
		up4 = BatchNorm(act=tf.nn.relu, is_train=self.train_bn, name='dbn4')(up4)

		up3 = Concat(concat_dim=3, name='concat3')([up4, conv4])
		up3 = DeConv2d(256, (4, 4), name='deconv3')(up3)
		up3 = BatchNorm(act=tf.nn.relu, is_train=self.train_bn, name='dbn3')(up3)

		up2 = Concat(concat_dim=3, name='concat2')([up3, conv3])
		up2 = DeConv2d(128, (4, 4), name='deconv2')(up2)
		up2 = BatchNorm(act=tf.nn.relu, is_train=self.train_bn, name='dbn2')(up2)

		up1 = Concat(concat_dim=3, name='concat1')([up2, conv2])
		up1 = DeConv2d(64, (4, 4),  name='deconv1')(up1)
		up1 = BatchNorm(act=tf.nn.relu, is_train=self.train_bn, name='dbn1')(up1)

		up0 = Concat(concat_dim=3, name='concat0')([up1, conv1])
		up0 = DeConv2d(64, (4, 4), name='deconv0')(up0)
		up0 = BatchNorm(act=tf.nn.relu, is_train=self.train_bn, name='dbn0')(up0)

		outs = Conv2d(3, (1, 1), act=tf.nn.sigmoid, name='out')(up0)
		return Model(inputs=inputs, outputs=outs, name="BN_U_Net")

	def criterion(self, x, y):
		return tf.sqrt(tl.cost.mean_squared_error(y, x), name='RMSE-Loss')


if __name__ == "__main__":
	image_size = 512
	test = TAMNet([None, image_size, image_size, 3])
	network = test.model
	print(network)
