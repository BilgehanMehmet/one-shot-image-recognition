from mnist import MNIST
from loader import Loader
import tensorflow as tf
import numpy as np
import cv2
from utils import plot,get_val_acc
import matplotlib.pyplot as plt


def rescale(X):
	X = X.reshape(-1,105,105)
	X.shape
	X_rescaled = []
	for idx in range(X.shape[0]):
		X_rescaled.append(cv2.resize(X[idx],(35,35),interpolation=cv2.INTER_AREA).reshape(35,35,1))
	return X_rescaled

	
def calc_one_shot_acc(nn,sess,test_data,val_data,n_trials_per_iter,n_trials):
	val_acc = 0
	one_shot_acc = 0
	for _ in range(n_trials // n_trials_per_iter): 
		test_imgs, train_imgs, labels = test_data.generateOneShotTrials(test_data.X_validation_rescaled,size=n_trials_per_iter)          
		val_acc += get_val_acc(test_imgs,train_imgs,labels,sess,nn,shape=(35,35))
		one_shot_test_imgs, one_shot_train_imgs, one_shot_labels = val_data.generateOneShotTrials(val_data.X_rescaled,size=n_trials_per_iter)          
		one_shot_acc += get_val_acc(one_shot_test_imgs,one_shot_train_imgs,one_shot_labels,sess,nn,shape=(35,35))
	return val_acc/ n_trials, one_shot_acc/ n_trials


def display_data(epoch,epoch_loss,epoch_acc,val_acc,one_shot_acc):
	print("Epoch:{}\t Epoch Loss:{}\t Train acc:{} \t Val acc:{}\t One Shot acc:{}".format(epoch,epoch_loss,epoch_acc,val_acc,one_shot_acc))
	print('{{"metric" : "Loss", "value":{}}}'.format(epoch_loss))
	print('{{"metric" : "Training Accuracy", "value":{}}}'.format(epoch_acc))
	print('{{"metric" : "Validation Accuracy", "value":{}}}'.format(val_acc))
	print('{{"metric" : "One-shot Accuracy", "value":{}}}'.format(one_shot_acc))


def train(nn,hyperparameters,train_data,test_data,val_data,epochs,batch_size,drawer_size):
	saver = tf.train.Saver()
	filepath = "/output/conv_siamese_mnist_model"
	seed = 1 
	n_trials = 400
	n_trials_per_iter = 10
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# saver.restore(sess,"/floyd/input/model/conv_siamese_model_40.ckpt")
		# print("conv_siamese_model_40.ckpt is restored.")
		for epoch in range(epochs+1):
			epoch_loss = 0
			epoch_acc = 0
			training_batch = train_data.get_training_pairs(batch_size= batch_size, drawer_size= drawer_size, seed=seed)
			seed += ((train_data.X.shape[0] * train_data.X.shape[1]) // training_batch[0][0].shape[0])       
			for counter, batch in enumerate(training_batch):
				X1 = np.asarray(rescale(batch[0]),dtype=np.float32).reshape(batch_size,35,35,1)
				X2 = np.asarray(rescale(batch[1]),dtype=np.float32).reshape(batch_size,35,35,1)
				Y  = batch[2]
				_, c, a= sess.run([nn.optimizer,nn.cost,nn.accuracy], feed_dict={
				nn.X:  X1,
				nn.X2: X2,
				nn.Y:  Y,
				nn.learning_rates[0]:hyperparameters[0]   * (0.99** epoch),#0.5
				nn.learning_rates[1]:hyperparameters[1] * (0.99** epoch),
				nn.learning_rates[2]:hyperparameters[2]  * (0.99** epoch),
				nn.learning_rates[3]:hyperparameters[3]  * (0.99** epoch),
				nn.learning_rates[4]:hyperparameters[4]  * (0.99** epoch),
				nn.momentums[0]:hyperparameters[5]  * (1.01** epoch),
				nn.momentums[1]:hyperparameters[6]  * (1.01** epoch),
				nn.momentums[2]:hyperparameters[7]  * (1.01** epoch),
				nn.momentums[3]:hyperparameters[8]  * (1.01** epoch),
				nn.momentums[4]:hyperparameters[9]  * (1.01** epoch)
				})
		epoch_acc += (a/len(training_batch))
		epoch_loss += (c/ len(training_batch))
		val_acc, one_shot_acc = calc_one_shot_acc(nn,sess,test_data,val_data,n_trials_per_iter,n_trials)
		display_data(epoch,epoch_loss,epoch_acc,val_acc,one_shot_acc)
		if ((epoch % 10 == 0) and epoch != 0):
			print("Saving the model...")
			saver.save(sess,filepath + "_" + str(epoch) + ".ckpt")
	return one_shot_acc


if __name__ == '__main__':
	tf.reset_default_graph()
	nn = MNIST(shape=(35,35))
	nn.forward_prop()
	nn.compute_cost(mode="l2_on")
	nn.optimize("Momentum")
	nn.accuracy()


	data = Loader(n_examples=100,mode="train_split",path = "/floyd/input/omniglot_dataset",normalise=True)
	test_data  = Loader(n_examples=20,mode=None,path="/floyd/input/val_set",normalise=True) 
	# data.X_rescaled = np.asarray(rescale(data.X),dtype=np.float32).reshape(964,-1,35,35,1)
	data.X_validation_rescaled = np.asarray(rescale(data.X_validation),dtype=np.float32).reshape(964,-1,35,35,1)
	test_data.X_rescaled = np.asarray(rescale(test_data.X),dtype=np.float32).reshape(352,-1,35,35,1)

	hyperparameters = [
	                    0.1,
	                    0.1,
	                    0.1,
	                    0.05,
	                    0.01,
	                    0.5,
	                    0.5,
	                    0.5,
	                    0.5,
	                    0.5
	          ]
	acc = train(nn,hyperparameters,data,data,test_data,epochs=50,batch_size=128,drawer_size=2)
	print("Final one shot accuracy:{}".format(acc))