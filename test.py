import numpy as np
import theano
import theano.tensor as T
import time
import cifar
import pickle
import random
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy import ndimage
from scipy.misc import imread, imsave, imresize, toimage


ar = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 2]
])
k = np.array([
    [1, -1, -1],
    [-1, 1, 0],
    [0, 1, 1]
])
car = ndimage.correlate(ar, k, mode='constant', cval=0)

print(ar * k)
print(k * ar)
print(np.dot(ar, k))
print(np.dot(k, ar))