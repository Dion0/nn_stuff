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
training_steps = 301
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
        scores = w.dot(x)
        margins = scores - scores[yi] + DELTA > 0
        margins[yi] = 0
        gxy = -(np.sum(margins) - 1) * w[yi]
        gx = (w.transpose() * margins).sum(axis=1) * bp_rate
        np.add(gx, gxy, gx)
        np.add(test, -gx, out=test)
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
    #grad += w * REG_DELTA
    grad[:][:-1] += w[:][:-1] * REG_DELTA
    return grad

def xgrad(x, y, w):
    return w[y]
    # scores = w.dot(x)
    # margins = scores - scores[y] + DELTA > 0
    # gwy = -(np.sum(margins) - 1) * x
    # grad = np.matmul(np.asmatrix(margins).transpose(), np.asmatrix(x))
    # grad[y] = gwy
    # return grad

def split_data(inp, out, batches_size = 256):
    spl_l = []
    ind = batches_size
    while ind < inp.shape[0]:
        spl_l.append(ind)
        ind += batches_size
    in_batch = np.split(inp, spl_l)
    out_batch = np.split(out, spl_l)
    return in_batch, out_batch



dict2 = cifar.unpickle('test_batch')
ev_in, ev_out = dict2['data'].astype(dtype=np.float64), dict2['labels']
del dict2


ev_in -= np.mean(ev_in, axis = 0)
ev_in /= (np.max(ev_in, axis=0) - np.min(ev_in, axis=0)) / 2
ev_in = np.column_stack((ev_in, np.ones(ev_in.shape[0])))

# some constants
SAMPLE_SIZE = len(ev_out)
INPUT_NUM = len(ev_in[0])
OUTPUT_NUM = IMG_CLASSES

#initialize weights with random values (with normal distribution for some reason)
w = rng.randn(OUTPUT_NUM, INPUT_NUM) * np.sqrt(2 / INPUT_NUM)

with open('fullw_nonorm_3', 'rb') as fw:
    w = pickle.load(fw)

print(w)

dict1 = cifar.unpickle("data_batch_1")
outputs, r_inputs = dict1['labels'], dict1['data'].astype(dtype=np.float64)
del dict1


