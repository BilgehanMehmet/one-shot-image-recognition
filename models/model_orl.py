import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from loader import Loader 
from omniglot import Omniglot
import os
import cv2
import utils


def generateOneShotTrials(X,size):
    '''
    Generates one shot trial for the data given where one one-shot trial is comparing one image against n-images.
    Therefore, comparing 20 different images against 20 images is 20 one-shot trials of each trial being 20-way.

    Arguments:
    data -- shape = (n_chars, n_drawers, 105, 105, 1)

    Returns:
    trainImgs -- shape  (20,105,105,1)
    testImgs -- shape (20,105,105,1)
    labels -- shape(1,20)
    '''
    chars = np.random.randint(low= 0,high= X.shape[0],size=size)
    drawers = np.random.randint(low=0 ,high=X[chars].shape[1],size=2) if X.shape[1] > 2 else np.random.permutation(2)
    testImgs = X[chars,drawers[0]]
    trainImgs = X[chars,drawers[1]]
    labels = chars
    return (testImgs,trainImgs,labels.reshape(1,size))


def calc_one_shot_acc(nn,sess,orl_data,n_trials_per_iter,n_trials):
    val_acc = 0
    one_shot_acc = 0
    for _ in range(n_trials // n_trials_per_iter): 
        test_imgs, train_imgs, labels = generateOneShotTrials(orl_data.X,size=n_trials_per_iter)          
        val_acc += utils.get_val_acc(test_imgs,train_imgs,labels,sess,nn,shape=(105,105))
        one_shot_test_imgs, one_shot_train_imgs, one_shot_labels = generateOneShotTrials(orl_data.one_shot_set,size=n_trials_per_iter)          
        one_shot_acc += utils.get_val_acc(one_shot_test_imgs,one_shot_train_imgs,one_shot_labels,sess,nn,shape=(105,105))
    return val_acc/ n_trials, one_shot_acc/ n_trials


def display_data(epoch,epoch_loss,epoch_acc,val_acc,one_shot_acc):
    print("Epoch:{}\t Epoch Loss:{}\t Train acc:{} \t Val acc:{}\t One Shot acc:{}".format(epoch,epoch_loss,epoch_acc,val_acc,one_shot_acc))
    print('{{"metric" : "Loss", "value":{}}}'.format(epoch_loss))
    print('{{"metric" : "Training Accuracy", "value":{}}}'.format(epoch_acc))
    print('{{"metric" : "Validation Accuracy", "value":{}}}'.format(val_acc))
    print('{{"metric" : "One-shot Accuracy", "value":{}}}'.format(one_shot_acc))


def train(hyperparameters,orl_data,epochs,batch_size,drawer_size):
    # CNN = Omniglot(shape=(105,105),cost_mode="contrastive_loss")
    CNN = Omniglot(shape=(105,105),cost_mode="binary_cross_entropy",batch_size= batch_size)
    CNN.forward_prop()
    CNN.compute_cost(mode = "l2_on")
    # CNN.compute_cost(mode = "l2_off",margin=1.)
    CNN.optimize(optimizer="Momentum")
    # CNN.optimize(optimizer="Adam",layer_wise=False,learning_rate=0.2)
    CNN.accuracy()
    saver = tf.train.Saver()
    filepath = "/output/conv_siamese_model_orl"
    seed = 1 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess,"/floyd/input/model/conv_siamese_model_orl_90.ckpt")
        # print("conv_siamese_model_orl_90.ckpt is restored.")
        for epoch in range(epochs+1):
            epoch_loss = 0
            epoch_acc = 0
            val_acc = 0
            one_shot_acc = 0
            n_trials_per_iter = 10
            n_trials = n_trials_per_iter * 40
            training_batch = orl_data.get_training_pairs(batch_size= batch_size, drawer_size= drawer_size, seed=seed)
            seed += ((orl_data.X_validation.shape[0] * orl_data.X_validation.shape[1]) // training_batch[0][0].shape[0])       
            for counter, batch in enumerate(training_batch):
                # print("In batch:{}".format(counter))
                X1 = batch[0]
                X2 = batch[1]
                Y  = batch[2]
                _, c, a= sess.run([CNN.optimizer,CNN.cost,CNN.accuracy], feed_dict={
                    CNN.X:  X1,
                    CNN.X2: X2,
                    CNN.Y:  Y,
                    CNN.learning_rates[0]:hyperparameters[0]   * (0.99** epoch),#0.5
                    CNN.learning_rates[1]:hyperparameters[1] * (0.99** epoch),
                    CNN.learning_rates[2]:hyperparameters[2]  * (0.99** epoch),
                    CNN.learning_rates[3]:hyperparameters[3]  * (0.99** epoch),
                    CNN.learning_rates[4]:hyperparameters[4]  * (0.99** epoch),
                    CNN.learning_rates[5]:hyperparameters[5]  * (0.99** epoch),#90% of prvs layer
                    CNN.momentums[0]:hyperparameters[6]  * (1.01** epoch),
                    CNN.momentums[1]:hyperparameters[7]  * (1.01** epoch),
                    CNN.momentums[2]:hyperparameters[8]  * (1.01** epoch),
                    CNN.momentums[3]:hyperparameters[9]  * (1.01** epoch),
                    CNN.momentums[4]:hyperparameters[10]  * (1.01** epoch),
                    CNN.momentums[5]:hyperparameters[11]  * (1.01** epoch)
                    
                })
                epoch_acc += (a/len(training_batch))
                epoch_loss += (c/ len(training_batch))
            val_acc, one_shot_acc = calc_one_shot_acc(CNN,sess,orl_data,n_trials_per_iter,n_trials)
            display_data(epoch,epoch_loss,epoch_acc,val_acc,one_shot_acc)
            if ((epoch % 10 == 0) and epoch != 0):
                print("Saving the model...")
                saver.save(sess,filepath + "_" + str(epoch) + ".ckpt")    


if __name__ == "__main__":
    orl = Loader(n_examples=10,mode="orl_split",path='/floyd/input/data',normalise=True)
    hyperparameters = [
                        0.05,
                        0.05,
                        0.05,
                        0.025,
                        0.005,
                        0.0005,
                        0.3,
                        0.3,
                        0.3,
                        0.3,
                        0.3,
                        0.3
              ]
    acc = train(hyperparameters,orl,epochs=300,batch_size=4,drawer_size=2)
