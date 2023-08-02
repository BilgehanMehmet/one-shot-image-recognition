import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from .base import Siamese

class Omniglot(Siamese):

	def __init__(self,shape=None,cost_mode=None,batch_size=None):
		assert batch_size is not None  
		self.network_size = 6
		self._FILTERS_CONV_1 = 64
		self._FILTERS_CONV_2 = 128
		self._FILTERS_CONV_3 = 128
		self._FILTERS_CONV_4 = 256
		self._FC_1_ACTIVATION = 4096
		self._FC_2_ACTIVATION = 1
		self._FLATTENED_CONV_LAYER = 256 * 6 * 6
		self._BATCH_SIZE = batch_size	
		self.lambdas = [
					0,
					0,
					0,
					0,
					0.3,
					0.3
		]

		# self.__dict__.update(dict(kwargs))
		super().__init__(shape= shape,cost_mode=cost_mode)
		

		with tf.variable_scope("model",reuse= tf.AUTO_REUSE):
			self.parameters = { 
				"W1" : tf.get_variable("W1",shape=[10,10,1,self._FILTERS_CONV_1],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1)),
				"W2" : tf.get_variable("W2",shape=[7,7,self._FILTERS_CONV_1,self._FILTERS_CONV_2],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1)),
				"W3" : tf.get_variable("W3",shape=[4,4,self._FILTERS_CONV_2,self._FILTERS_CONV_3],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1)),
				"W4" : tf.get_variable("W4",shape=[4,4,self._FILTERS_CONV_3,self._FILTERS_CONV_4],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1)),
				"W5" : tf.get_variable("W5",shape=[self._FC_1_ACTIVATION,self._FLATTENED_CONV_LAYER],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1)),
				"W6" : tf.get_variable("W6",shape=[self._FC_2_ACTIVATION,self._FC_1_ACTIVATION],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1)),

				"b1" : tf.get_variable("b1",shape=[1,1,1,self._FILTERS_CONV_1],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1)),
				"b2" : tf.get_variable("b2",shape=[1,1,1,self._FILTERS_CONV_2],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1)),
				"b3" : tf.get_variable("b3",shape=[1,1,1,self._FILTERS_CONV_3],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1)),
				"b4" : tf.get_variable("b4",shape=[1,1,1,self._FILTERS_CONV_4],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1)),
				"b5" : tf.get_variable("b5",shape=[self._FC_1_ACTIVATION,1],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1)),
				"b6" : tf.get_variable("b6",shape=[self._FC_2_ACTIVATION,1],dtype=tf.float32,initializer= tf.contrib.layers.xavier_initializer(seed=1))
				}

			self._var_list = [
					[self.parameters["W1"],self.parameters["b1"]],
					[self.parameters["W2"],self.parameters["b2"]],
				    [self.parameters["W3"],self.parameters["b3"]],
				    [self.parameters["W4"],self.parameters["b4"]],
				    [self.parameters["W5"],self.parameters["b5"]],
				    [self.parameters["W6"],self.parameters["b6"]]
				    		]

	def __str__(self):
		return "Twin_1_dist_layer:{}\nTwin_2_dist_layer:{}\nlogits:{}\ncost:{}\noptimizer:{}\naccuracy:{}".format(
			self.twin1_fc_1, 
			self.twin2_fc_1,
			self.logits,
			self.cost,
			self.optimizer,
			self.accuracy
			)

	def _network(self,X):
		assert self.network_size == 6, "Network size should be of length 6, or change the network architecture"
		# CONV1 -> POOL1 -> CONV2 -> POOL2 -> CONV3 -> POOL3 -> CONV4 -> -> FLATTEN -> FC1(SIGMOID)
		# return FC1
		self.conv1 = self._conv_layer(X,weight="W1",bias="b1")
		self.pool1 = self._max_pool_layer(self.conv1)
		# print(self.pool1)

		self.conv2 = self._conv_layer(self.pool1,weight="W2",bias="b2")
		# print(self.conv2)
		self.pool2 = self._max_pool_layer(self.conv2)
		# print(self.pool2)

		self.conv3 = self._conv_layer(self.pool2,weight="W3",bias="b3")
		self.pool3 = self._max_pool_layer(self.conv3)
		# print(self.conv3)
		# print(self.pool3)

		self.conv4 = self._conv_layer(self.pool3,weight="W4",bias="b4")
		# print(self.conv4)

		self.flattened = tf.transpose(tf.contrib.layers.flatten(inputs = self.conv4))
		# print(self.flattened)

		if self.cost_mode == "contrastive_loss":
			# return tf.divide(self.flattened,tf.norm(self.flattened,axis=1,keepdims=True))  # axis=1 means normalise between features, 
			# axis=0 means normalise embedding pixels relative to embedding only
			# or try tf.nn.l2_normalize(self.flattened,axis=0)
			return tf.nn.l2_normalize(self.flattened,axis=1)
		elif self.cost_mode == "binary_cross_entropy":
			return tf.nn.sigmoid(self._fc_layer(X=self.flattened ,weight="W5", bias="b5"))
		# return tf.nn.relu(self.fc_layer(X=self.flattened ,weight="W5", bias="b5"))
		


	def forward_prop(self):
		self.twin1_fc_1 = self._network(X = self.X)
		self.twin2_fc_1 = self._network(X = self.X2)
		self.diff = tf.subtract(self.twin1_fc_1,self.twin2_fc_1)
		# logits = tf.nn.sigmoid(self.fc_layer(X=diff,weight="W6",bias="b6"))	
		if self.cost_mode == "binary_cross_entropy":
			self.logits = self._fc_layer(X=tf.abs(self.diff),weight="W6",bias="b6")
		elif self.cost_mode == "contrastive_loss":
			self.logits = tf.norm(self.diff,axis=0,keepdims= True)

	def _get_frobenius_norm(self):
		if self.cost_mode == "binary_cross_entropy":
			return (
				((self.lambdas[0]/self._BATCH_SIZE)*tf.nn.l2_loss(self.parameters["W1"])) + 
				((self.lambdas[1]/self._BATCH_SIZE)*tf.nn.l2_loss(self.parameters["W2"])) + 
				((self.lambdas[2]/self._BATCH_SIZE)*tf.nn.l2_loss(self.parameters["W3"])) + 
				((self.lambdas[3]/self._BATCH_SIZE)*tf.nn.l2_loss(self.parameters["W4"])) + 
				((self.lambdas[4]/self._BATCH_SIZE)*tf.nn.l2_loss(self.parameters["W5"])) + 
				((self.lambdas[5]/self._BATCH_SIZE)*tf.nn.l2_loss(self.parameters["W6"]))
					)
		elif self.cost_mode == "contrastive_loss":
			return (
				((self.lambdas[0]/self._BATCH_SIZE)*tf.nn.l2_loss(self.parameters["W1"])) + 
				((self.lambdas[1]/self._BATCH_SIZE)*tf.nn.l2_loss(self.parameters["W2"])) + 
				((self.lambdas[2]/self._BATCH_SIZE)*tf.nn.l2_loss(self.parameters["W3"])) + 
				((self.lambdas[3]/self._BATCH_SIZE)*tf.nn.l2_loss(self.parameters["W4"]))
					)


	def compute_cost(self,mode=None,margin=None):
		if self.cost_mode == "contrastive_loss":
			assert margin is not None 
			self.margin = margin
		assert mode != None, "Mode should either be 'l2_on' or 'l2_off'"
		assert self.network_size == 6, "Network size should be of length 6, or change the computation of cost"
		if self.cost_mode == "binary_cross_entropy":
			self.cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels= self.Y)
		elif self.cost_mode == "contrastive_loss":
			self.cost  = 0.5 * ((1-self.Y) * tf.square((self.logits)) + (self.Y * tf.square(tf.maximum(
					0.,self.margin - tf.sqrt(tf.math.add(tf.square(self.logits),1e-6)))))) # commented tf.math.add(logits,1e-6)
		if mode == "l2_on":
			self.cost += self._get_frobenius_norm()
		self.cost = tf.reduce_mean(self.cost)
		


