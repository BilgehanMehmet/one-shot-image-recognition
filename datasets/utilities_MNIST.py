import numpy as np
import mnist
import matplotlib.pyplot as plt



def test_pairs(x_test,y_test):
    digits_test={}
    for digit in range(10):
        idx_test = np.argwhere(y_test==digit)
        # print(idx_zero)
        # print(y[idx_zero[1,0],idx_zero[1,1]])
        # temp = x[:,idx_zero[2,1]].reshape(28,28)
        # plt.imshow(temp)
        # plt.show()
        digits_test["x" + str(digit)] = np.zeros([x_test.shape[0],idx_test.shape[0]])
    #     digits["y" + str(digit)] = np.zeros([1,idx.shape[0]])
    #     digits_test["x" + str(digit)] = np.zeros([x.shape[0],idx.shape[0]])
        for i in range(idx_test.shape[0]):
    #         digits["y" + str(digit)][0,i] = y[idx[i,0],idx[i,1]]
            digits_test["x" + str(digit)][:,i] = x_test[:,idx_test[i,1]]

    input_pairs_test = np.concatenate((digits_test["x0"][:,:200],digits_test["x0"][:,200:400]),axis=0)
    labels_pairs_test = np.ones([1,200]) 

    input_pairs_test = np.concatenate((input_pairs_test,np.concatenate((digits_test["x1"][:,:200],digits_test["x1"][:,200:400]),axis=0)),axis=1)
    labels_pairs_test = np.concatenate((labels_pairs_test,np.ones([1,200])),axis=1) 

    input_pairs_test = np.concatenate((input_pairs_test,np.concatenate((digits_test["x0"][:,400:800],digits_test["x1"][:,400:800]),axis=0)),axis=1)
    labels_pairs_test = np.concatenate((labels_pairs_test,np.zeros([1,400])),axis=1) 


    input_pairs_test = np.concatenate((input_pairs_test,np.concatenate((digits_test["x2"][:,:200],digits_test["x2"][:,200:400]),axis=0)),axis=1)
    labels_pairs_test = np.concatenate((labels_pairs_test,np.ones([1,200])),axis=1)

    input_pairs_test = np.concatenate((input_pairs_test,np.concatenate((digits_test["x3"][:,:200],digits_test["x3"][:,200:400]),axis=0)),axis=1)
    labels_pairs_test = np.concatenate((labels_pairs_test,np.ones([1,200])),axis=1) 

    input_pairs_test = np.concatenate((input_pairs_test,np.concatenate((digits_test["x2"][:,400:800],digits_test["x3"][:,400:800]),axis=0)),axis=1)
    labels_pairs_test = np.concatenate((labels_pairs_test,np.zeros([1,400])),axis=1) 



    input_pairs_test = np.concatenate((input_pairs_test,np.concatenate((digits_test["x4"][:,:200],digits_test["x4"][:,200:400]),axis=0)),axis=1)
    labels_pairs_test = np.concatenate((labels_pairs_test,np.ones([1,200])),axis=1)


    input_pairs_test = np.concatenate((input_pairs_test,np.concatenate((digits_test["x5"][:,:200],digits_test["x5"][:,200:400]),axis=0)),axis=1)
    labels_pairs_test = np.concatenate((labels_pairs_test,np.ones([1,200])),axis=1) 

    input_pairs_test = np.concatenate((input_pairs_test,np.concatenate((digits_test["x4"][:,400:800],digits_test["x5"][:,400:800]),axis=0)),axis=1)
    labels_pairs_test = np.concatenate((labels_pairs_test,np.zeros([1,400])),axis=1) 



    input_pairs_test = np.concatenate((input_pairs_test,np.concatenate((digits_test["x6"][:,:200],digits_test["x6"][:,200:400]),axis=0)),axis=1)
    labels_pairs_test = np.concatenate((labels_pairs_test,np.ones([1,200])),axis=1)


    input_pairs_test = np.concatenate((input_pairs_test,np.concatenate((digits_test["x7"][:,:200],digits_test["x7"][:,200:400]),axis=0)),axis=1)
    labels_pairs_test = np.concatenate((labels_pairs_test,np.ones([1,200])),axis=1) 

    input_pairs_test = np.concatenate((input_pairs_test,np.concatenate((digits_test["x6"][:,400:800],digits_test["x7"][:,400:800]),axis=0)),axis=1)
    labels_pairs_test = np.concatenate((labels_pairs_test,np.zeros([1,400])),axis=1) 



    input_pairs_test = np.concatenate((input_pairs_test,np.concatenate((digits_test["x8"][:,:200],digits_test["x8"][:,200:400]),axis=0)),axis=1)
    labels_pairs_test = np.concatenate((labels_pairs_test,np.ones([1,200])),axis=1)

    input_pairs_test = np.concatenate((input_pairs_test,np.concatenate((digits_test["x9"][:,:200],digits_test["x9"][:,200:400]),axis=0)),axis=1)
    labels_pairs_test = np.concatenate((labels_pairs_test,np.ones([1,200])),axis=1) 

    input_pairs_test = np.concatenate((input_pairs_test,np.concatenate((digits_test["x8"][:,400:800],digits_test["x9"][:,400:800]),axis=0)),axis=1)
    labels_pairs_test = np.concatenate((labels_pairs_test,np.zeros([1,400])),axis=1) 

    return (input_pairs_test,labels_pairs_test)


def getBatch(minibatch_size,shuffled_input_pairs,shuffled_labels):
    batch = []
#     minibatch_size = 128

    remainder = shuffled_input_pairs.shape[1] % minibatch_size
    for index in range(0,shuffled_input_pairs.shape[1] - remainder,minibatch_size):
        batch.append((shuffled_input_pairs[:,index:index+minibatch_size],shuffled_labels[:,index:index+minibatch_size]))
        
    batch.append((shuffled_input_pairs[:,-remainder:],shuffled_labels[:,-remainder:]))    

    return batch


def normalise(x_):
    return np.divide(x_,np.var(x_))

def get_training_pairs(x,y):
    digits={}
    for digit in range(10):
        idx = np.argwhere(y==digit)
        # print(idx_zero)
        # print(y[idx_zero[1,0],idx_zero[1,1]])
        # temp = x[:,idx_zero[2,1]].reshape(28,28)
        # plt.imshow(temp)
        # plt.show()
        digits["x" + str(digit)] = np.zeros([x.shape[0],idx.shape[0]])
        digits["y" + str(digit)] = np.zeros([1,idx.shape[0]])
    #     digits_test["x" + str(digit)] = np.zeros([x.shape[0],idx.shape[0]])
        for i in range(idx.shape[0]):
            digits["y" + str(digit)][0,i] = y[idx[i,0],idx[i,1]]
            digits["x" + str(digit)][:,i] = x[:,idx[i,1]]
        
    input_pairs = np.concatenate((digits["x0"][:,:1350],digits["x0"][:,1350:2700]),axis=0)
    labels_pairs = np.ones([1,1350]) 

    input_pairs = np.concatenate((input_pairs,np.concatenate((digits["x1"][:,:1350],digits["x1"][:,1350:2700]),axis=0)),axis=1)
    labels_pairs = np.concatenate((labels_pairs,np.ones([1,1350])),axis=1) 

    input_pairs = np.concatenate((input_pairs,np.concatenate((digits["x0"][:,2700:5400],digits["x1"][:,2700:5400]),axis=0)),axis=1)
    labels_pairs = np.concatenate((labels_pairs,np.zeros([1,2700])),axis=1) 




    input_pairs = np.concatenate((input_pairs,np.concatenate((digits["x2"][:,:1350],digits["x2"][:,1350:2700]),axis=0)),axis=1)
    labels_pairs = np.concatenate((labels_pairs,np.ones([1,1350])),axis=1) 

    input_pairs = np.concatenate((input_pairs,np.concatenate((digits["x3"][:,:1350],digits["x3"][:,1350:2700]),axis=0)),axis=1)
    labels_pairs = np.concatenate((labels_pairs,np.ones([1,1350])),axis=1) 

    input_pairs = np.concatenate((input_pairs,np.concatenate((digits["x2"][:,2700:5400],digits["x3"][:,2700:5400]),axis=0)),axis=1)
    labels_pairs = np.concatenate((labels_pairs,np.zeros([1,2700])),axis=1) 




    input_pairs = np.concatenate((input_pairs,np.concatenate((digits["x4"][:,:1350],digits["x4"][:,1350:2700]),axis=0)),axis=1)
    labels_pairs = np.concatenate((labels_pairs,np.ones([1,1350])),axis=1) 

    input_pairs = np.concatenate((input_pairs,np.concatenate((digits["x5"][:,:1350],digits["x5"][:,1350:2700]),axis=0)),axis=1)
    labels_pairs = np.concatenate((labels_pairs,np.ones([1,1350])),axis=1) 

    input_pairs = np.concatenate((input_pairs,np.concatenate((digits["x4"][:,2700:5400],digits["x5"][:,2700:5400]),axis=0)),axis=1)
    labels_pairs = np.concatenate((labels_pairs,np.zeros([1,2700])),axis=1) 



    input_pairs = np.concatenate((input_pairs,np.concatenate((digits["x6"][:,:1350],digits["x6"][:,1350:2700]),axis=0)),axis=1)
    labels_pairs = np.concatenate((labels_pairs,np.ones([1,1350])),axis=1) 

    input_pairs = np.concatenate((input_pairs,np.concatenate((digits["x7"][:,:1350],digits["x7"][:,1350:2700]),axis=0)),axis=1)
    labels_pairs = np.concatenate((labels_pairs,np.ones([1,1350])),axis=1) 

    input_pairs = np.concatenate((input_pairs,np.concatenate((digits["x6"][:,2700:5400],digits["x7"][:,2700:5400]),axis=0)),axis=1)
    labels_pairs = np.concatenate((labels_pairs,np.zeros([1,2700])),axis=1) 



    input_pairs = np.concatenate((input_pairs,np.concatenate((digits["x8"][:,:1350],digits["x8"][:,1350:2700]),axis=0)),axis=1)
    labels_pairs = np.concatenate((labels_pairs,np.ones([1,1350])),axis=1) 

    input_pairs = np.concatenate((input_pairs,np.concatenate((digits["x9"][:,:1350],digits["x9"][:,1350:2700]),axis=0)),axis=1)
    labels_pairs = np.concatenate((labels_pairs,np.ones([1,1350])),axis=1) 

    input_pairs = np.concatenate((input_pairs,np.concatenate((digits["x8"][:,2700:5400],digits["x9"][:,2700:5400]),axis=0)),axis=1)
    labels_pairs = np.concatenate((labels_pairs,np.zeros([1,2700])),axis=1) 

    return (input_pairs,labels_pairs)
