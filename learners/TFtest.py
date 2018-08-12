from tensorflow.examples.tutorials.mnist import input_data
from scipy.io import loadmat
import numpy as np
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#res = mnist.test.next_batch(batch_size=100)
#print(res[0].shape)
#print(res[1].shape)


trainfile = "datasets/housenumbers/train_32x32.mat"
trainset = loadmat(trainfile)
X = trainset['X']
y = trainset['y']
print(X.shape)
print(y.shape)
yvec = np.zeros((y.shape[0], len(np.unique(y))))
yvec[np.arange(y.shape[0]), (y-1).flatten()] = 1
print(yvec)