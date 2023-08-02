
'''
Deep Siamese Network Implementation in Tensorflow 

File: Siamese_Tensorflow.py
Author: Mehmet Bilgehan Bezcioglu
Date modified: 17/10/2018

'''

import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
import os 

class Siamese(object):
	
	def __init__(self):
		pass


	def create_placeholders(self):
		pass


	def initialize_parameters(self):
		pass


	def forward_prop(self):
		pass


	def compute_cost(self):
		pass


	def optimizer(self):
		pass

	def model(self):
		pass


if __name__ == "__main__": 
	# np.fromfile()
	os.chdir('./Siamese_github')
	print(os.getcwd())

	# pretrained_model = np.fromfile('/Users/mehmetbezcioglu/Documents/SEMESTER1/DeepLearning/Siamese_github/embed.txt')
	# pretrained_model = pretrained_model.reshape(10000,-1)
	# print(pretrained_model.shape)

	tf.train.Saver().restore(sess,'./model')