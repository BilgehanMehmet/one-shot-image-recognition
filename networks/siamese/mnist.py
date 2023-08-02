import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from .base import Siamese

class MNIST(Siamese):

	def __init__(self, shape=None):
		self.network_size = 5
		self._FILTERS_CONV_1 = 96
		self._FILTERS_CONV_2 = 256
		self._FILTERS_CONV_3 = 256
		self._FC_1_ACTIVATION = 2048
		self._FC_2_ACTIVATION = 1
		self._FLATTENED_CONV_LAYER = 256 * 4 * 4
		self._BATCH_SIZE = 128	
		self.lambdas = [
					0,
					0,
					0,
					0,
					0.05
		]

		# self.__dict__.update(dict(kwargs))
		super().__init__(shape= shape)
		

		with tf.variable_scope("model",reuse= tf.AUTO_REUSE):
			self.parameters = { 
				"W1" : tf.get_variable("W1",shape=[3,3,1,self._FILTERS_CONV_1],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1)),
				"W2" : tf.get_variable("W2",shape=[5,5,self._FILTERS_CONV_1,self._FILTERS_CONV_2],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1)),
				"W3" : tf.get_variable("W3",shape=[3,3,self._FILTERS_CONV_2,self._FILTERS_CONV_3],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1)),
				"W4" : tf.get_variable("W5",shape=[self._FC_1_ACTIVATION,self._FLATTENED_CONV_LAYER],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1)),
				"W5" : tf.get_variable("W6",shape=[self._FC_2_ACTIVATION,self._FC_1_ACTIVATION],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1)),

				"b1" : tf.get_variable("b1",shape=[1,1,1,self._FILTERS_CONV_1],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1)),
				"b2" : tf.get_variable("b2",shape=[1,1,1,self._FILTERS_CONV_2],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1)),
				"b3" : tf.get_variable("b3",shape=[1,1,1,self._FILTERS_CONV_3],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1)),
				"b4" : tf.get_variable("b5",shape=[self._FC_1_ACTIVATION,1],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1)),
				"b5" : tf.get_variable("b6",shape=[self._FC_2_ACTIVATION,1],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1))
				}

			self._var_list = [
					[self.parameters["W1"],self.parameters["b1"]],
					[self.parameters["W2"],self.parameters["b2"]],
				    [self.parameters["W3"],self.parameters["b3"]],
				    [self.parameters["W4"],self.parameters["b4"]],
				    [self.parameters["W5"],self.parameters["b5"]]
				    		]



	def __str__(self):
		# return "<MNIST:{}>".format([layer for layer in self.__dict__ if layer[0].isupper() and 
		# 	not layer.startswith('_')])
		return "<MNIST:\n CONV_1: {}\n POOL_1: {}\n CONV_2: {}\n POOL_2: {}\n CONV_3: {}\n FLATTENED:{}\n>".format(self.conv1, self.pool1, self.conv2, self.pool2, self.conv3, self.flattened)

	def _network(self,X):
		# CONV1 -> POOL1 -> CONV2 -> POOL2 -> CONV3 -> -> FLATTEN -> FC1(RELU)

		assert self.network_size == 5, "Network size should be of length 5, or change the network architecture"

		self.conv1 = self._conv_layer(X,weight="W1",bias="b1")
		# print(self.conv1)
		self.pool1 = self._max_pool_layer(self.conv1)
		# print(self.pool1)

		self.conv2 = self._conv_layer(self.pool1,weight="W2",bias="b2")
		# print(self.conv2)
		self.pool2 = self._max_pool_layer(self.conv2)
		# print(self.pool2)

		self.conv3 = self._conv_layer(self.pool2,weight="W3",bias="b3")
		self.flattened = tf.transpose(tf.contrib.layers.flatten(inputs = self.conv3))
		# print(self.flattened)

		# return tf.nn.sigmoid(self.fc_layer(X=self.flattened ,weight="W5", bias="b5"))
		return tf.nn.relu(self._fc_layer(X=self.flattened ,weight="W4", bias="b4"))
		

	def forward_prop(self):
		self.twin1_fc_1 = self._network(X = self.X)
		self.twin2_fc_1 = self._network(X = self.X2)
		diff = tf.abs(tf.subtract(self.twin1_fc_1,self.twin2_fc_1))
		# logits = tf.nn.sigmoid(self.fc_layer(X=diff,weight="W6",bias="b6"))	
		self.logits = self._fc_layer(X=diff,weight="W5",bias="b5")


	def compute_cost(self,mode=None):
		assert mode != None, "Mode should either be 'l2_on' or 'l2_off'"
		if mode == "l2_on":
			assert self.network_size == 5, "Network size should be of length 5, or change the computation of cost"
			self.cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels= self.Y)  +
							((self.lambdas[0]/self._BATCH_SIZE)*tf.nn.l2_loss(self.parameters["W1"])) + 
							((self.lambdas[1]/self._BATCH_SIZE)*tf.nn.l2_loss(self.parameters["W2"])) + 
							((self.lambdas[2]/self._BATCH_SIZE)*tf.nn.l2_loss(self.parameters["W3"])) + 
							((self.lambdas[3]/self._BATCH_SIZE)*tf.nn.l2_loss(self.parameters["W4"])) + 
							((self.lambdas[4]/self._BATCH_SIZE)*tf.nn.l2_loss(self.parameters["W5"]))
							)
		elif mode == "l2_off":
			self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels= self.Y))


