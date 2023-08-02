import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Siamese():

	def __init__(self,shape,cost_mode):
		assert shape is not None, "Assign shape for the inputs."
		assert ((cost_mode == "binary_cross_entropy") or (cost_mode == "contrastive_loss")), "Cost mode should be binary_cross_entropy or contrastive_loss."
		self.cost_mode = cost_mode
		self.X = tf.placeholder(dtype=tf.float32, shape=(None,) + shape + (1,), name="X")
		self.X2 = tf.placeholder(dtype=tf.float32, shape=(None,) + shape + (1,), name="X2")
		self.Y = tf.placeholder(dtype=tf.float32, shape=(1,None), name="Y")
		self.learning_rates = [tf.placeholder(dtype=tf.float32, shape=[],name="l" + str(i)) for i in range(self.network_size)]
		self.momentums = [tf.placeholder(dtype=tf.float32, shape=[],name = "m" + str(i)) for i in range(self.network_size)]


	def _conv_layer(self,X,weight,bias,strides = [1,1,1,1],padding = "VALID"):
		return tf.nn.relu(
					tf.add(
						tf.nn.conv2d(
								input = X,
								filter = self.parameters[weight],
								strides = strides,
								padding = padding
								),
						self.parameters[bias]
						)
					)
	

	def _max_pool_layer(self,X,ksize= [1,2,2,1],strides= [1,2,2,1]):
		return tf.nn.max_pool(
					value= X,
					ksize= ksize,
					strides= strides,
					padding= "VALID"
				)


	def _fc_layer(self,X,weight,bias):
		return tf.add(
					tf.matmul(
							self.parameters[weight],
							X
							),
					self.parameters[bias]
					)


	def optimize(self,optimizer,layer_wise=True,learning_rate=0.001,momentum=0.5):
		if not layer_wise:
			if optimizer == "Adam":
				self.optimizer =  tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(self.cost)
			elif optimizer == "Momentum":
				self.optimizer =  tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum).minimize(self.cost)
			elif optimizer == "GradientDescent":
				self.optimizer =  tf.train.GradientDescentOptimizer(learning_rate= learning_rate).minimize(self.cost)
		else:
			self._gradients = self._get_gradients()
			if optimizer == "Adam":
				self.optimizer =   [tf.train.AdamOptimizer(learning_rate = self.learning_rates[i]).apply_gradients(
					list(zip(self._gradients[2*i:2*i+2],self._var_list[i]))
					) for i in range(self.network_size)]
			elif optimizer == "Momentum":
				self.optimizer =   [tf.train.MomentumOptimizer(learning_rate = self.learning_rates[i],
					momentum = self.momentums[i]).apply_gradients(
					list(zip(self._gradients[2*i:2*i+2],self._var_list[i])) ) for i in range(self.network_size)]


	def accuracy(self):
		if self.cost_mode == "binary_cross_entropy":
			self.probabilities = tf.nn.sigmoid(self.logits)
			self.predicted_labels = self.logits > 0.5
			self.accuracy = tf.reduce_mean(tf.cast((tf.math.equal(tf.cast(self.predicted_labels,tf.float32),self.Y)),tf.float32))

		elif self.cost_mode == "contrastive_loss":
			self.probabilities = self.logits
			self.predicted_labels = tf.cast(tf.maximum(tf.subtract(self.logits,self.margin),0.),dtype=tf.bool)
			self.accuracy = tf.reduce_mean(tf.cast((tf.math.equal(
				tf.cast(
					self.predicted_labels,
					tf.int32
					),
				tf.cast(
					self.Y,
					tf.int32
					)
				)
			),tf.float32)
			)


	def _get_gradients(self):
		return tf.gradients(ys = self.cost, xs= [item for sublist in self._var_list for item in sublist])