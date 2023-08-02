import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# import Siamese_Dataset_Loader as Dataset 
import sys
sys.path.append('..')
from networks.siamese import MNIST
from bin import utils
import os
import cv2

def train(nn,hyperparameters,train_data,test_data,val_data,epochs,batch_size,drawer_size):
	saver = tf.train.Saver()
	filepath = "/output/conv_siamese_model_mnist"
	seed = 1
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(epochs+1):
			pass
		

if __name__ == '__main__':
	# build the model
	tf.reset_default_graph()
	nn = MNIST(shape=(35,35))
	nn.forward_prop()
	nn.compute_cost(mode="l2_on")
	nn.accuracy()

	# train the model

