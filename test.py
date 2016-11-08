import numpy as np
import theano
import theano.tensor as T
import cifar
rng = np.random


# generate a dataset: D = (input_values, target_class)
#D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
# forward pass
W = np.random.randint(-4, 5, (5, 7))
X = np.random.randint(-4, 5, (7, 3))
D = W.dot(X)

# now suppose we had the gradient on D from above in the circuit
dD = np.random.randn(*D.shape) # same shape as D
dW = dD.dot(X.T) #.T gives the transpose of the matrix
dX = W.T.dot(dD)

print(W)
print(X)
print(D)

print(dD)
print(dW)
print(dX)