import numpy as np
import theano
import theano.tensor as T
import time
import cifar
import pickle
import random


rng = np.random
#rng.seed(123)

IMG_CLASSES = 10
REG_DELTA = 0.005
DELTA = 1.0
training_steps = 10000
learning_rate = 0.0001

decode_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


#svm loss
def loss(x, y, w):
    scores = w.dot(x)
    margins = np.maximum(0, scores - scores[y] + DELTA)
    margins[y] = 0
    return np.sum(margins) + np.sum(np.square(w)) * REG_DELTA / 2.0

#all losses \ o /
def total_loss(x, y, w):
    loss_sum = 0
    for inp, out in zip(x, y):
        loss_sum += loss(inp, out, w)
    return loss_sum

def evaluate(inputs, outputs, w):
    cnt = 0
    for inp, out in zip(inputs, outputs):
        tmp = w.dot(inp)
        ans = np.argmax(tmp)
        #print(ans, ' - ', out)
        if out != ans:
            cnt += 1
    print(cnt)
    return cnt

#backprob and stuff
def sample_image(ii, yi, w, cycles = 10000, bp_rate = 0.01, test = None):
    if test == None:
        test = np.random.randn(3073) * 0.01
    test[:][-1] = 1
    ans = [0] * IMG_CLASSES
    ans[yi] = 1
    for i in range(cycles):
        gx = (w.transpose() * ans).sum(axis=1) * bp_rate
        np.add(test, gx, out=test)
    tmp = w.dot(test)
    test = test + 1.0
    np.multiply(test, 128, test)
    #test = test.astype(dtype=np.uint8)
    cifar.save_img(test, 'wow{}_{}.png'.format(ii,decode_list[yi]))

def wgrad(x, y, w):
    scores = w.dot(x)
    margins = scores - scores[y] + DELTA > 0
    gwy = -(np.sum(margins) - 1) * x
    grad = np.matmul(np.asmatrix(margins).transpose(), np.asmatrix(x))
    grad[y] = gwy
    grad[:-1] += w[:-1] * REG_DELTA
    return grad

def xgrad(x, y, w):
    return w[y]
    # scores = w.dot(x)
    # margins = scores - scores[y] + DELTA > 0
    # gwy = -(np.sum(margins) - 1) * x
    # grad = np.matmul(np.asmatrix(margins).transpose(), np.asmatrix(x))
    # grad[y] = gwy
    # return grad


# loading cifar batch 1
dict1 = cifar.unpickle("data_batch_1")
dict2 = cifar.unpickle('test_batch')

outputs, r_inputs = dict1['labels'], dict1['data'].astype(dtype=np.float64)
ev_in, ev_out = dict2['data'].astype(dtype=np.float64), dict2['labels']

# normalize data
r_inputs -= np.mean(r_inputs, axis = 0)
r_inputs /= (np.max(r_inputs, axis=0) - np.min(r_inputs, axis=0)) / 2
ev_in -= np.mean(ev_in, axis = 0)
ev_in /= (np.max(ev_in, axis=0) - np.min(ev_in, axis=0)) / 2
#r_inputs = r_inputs / 128.0 - 1.0
#ev_in = ev_in / 128.0 - 1.0
# adding a column of ones for convinience (no need for 'b' vectors)
inputs = np.column_stack((r_inputs, np.ones(r_inputs.shape[0])))
ev_in = np.column_stack((ev_in, np.ones(ev_in.shape[0])))

# some constants
SAMPLE_SIZE = len(outputs)
INPUT_NUM = len(inputs[0])
OUTPUT_NUM = IMG_CLASSES

#initialize weights with random values (with normal distribution for some reason)
w = rng.randn(OUTPUT_NUM, INPUT_NUM) * np.sqrt(2/INPUT_NUM)

with open('fullw_nonorm', 'rb') as fw:
    w = pickle.load(fw)

#3500 7200
#3408 7127
#3403 7056
#3627 6921 // wtf? tendency to perform worse on the training set than on the test set?
#3876 6803 //shrug
#4131 6799
#4188 6782


for i in range(100):
    for x, y in zip(inputs, outputs):
        scores = w.dot(x)
        gw = wgrad(x, y, w)
        w -= gw * learning_rate
    print(total_loss(inputs, outputs, w))

with open('fullw_nonorm', 'wb') as fw:
    pickle.dump(w, fw)

w = rng.randn(OUTPUT_NUM, INPUT_NUM)
ss = 0
for i in range(10):
    ss += loss(inputs[i], outputs[i], w)

print(ss / 10.0)

evaluate(inputs, outputs, w)
evaluate(ev_in, ev_out, w)

for i in range(IMG_CLASSES):
    sample_image(150, i, w, 1050, 0.0001)



