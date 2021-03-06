import numpy as np
import cifar
import pickle
import matplotlib.pyplot as plt


rng = np.random
#rng.seed(123)

IMG_CLASSES = 10
DELTA = 1.0
training_steps = 2001
reg = 5e-1
step_size = 1e-0

H = 32

W1 = 0.01 * np.random.randn(3073, IMG_CLASSES)

decode_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def evaluate(inputs, outputs, w):
    scores = np.dot(inputs, w)
    ans = scores.argmax(axis=1)
    np.equal(ans, outputs, ans)
    return np.sum(ans)

def sample_images(ee, W1, cycles = 1000, bp_rate = 0.01, test = None):
    pass
    if test == None:
        test = np.random.randn(10, 3073) * 0.01
    test[:][-1] = 1
    for i in range(cycles):
        scores = np.dot(test, W1)
        exp_scores = np.exp(scores)
        dscores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        dscores[range(10), range(10)] -= 1
        dscores /= 10
        dX = np.dot(dscores, W1.T)
        test += -bp_rate * dX
    for j in range(test.shape[0]):
        cifar.save_img(test[j], '{}_{}_out.png'.format(ee, j))


dict2 = cifar.unpickle('test_batch')
ev_in, ev_out = dict2['data'].astype(dtype=np.float64), dict2['labels']
del dict2


ev_in -= np.mean(ev_in, axis = 0)
ev_in /= (np.max(ev_in, axis=0) - np.min(ev_in, axis=0) + 1e-5) / 2
ev_in = np.column_stack((ev_in, np.ones(ev_in.shape[0])))


for batch_num in range(5):
    print('batch #', batch_num + 1)
    # load data
    dict1 = cifar.unpickle("data_batch_" + str(batch_num + 1))
    outputs, r_inputs = dict1['labels'], dict1['data'].astype(dtype=np.float64)
    del dict1

    # normalize data
    r_inputs -= np.mean(r_inputs, axis=0)
    r_inputs /= (np.max(r_inputs, axis=0) - np.min(r_inputs, axis=0)) / 2
    inputs = np.column_stack((r_inputs, np.ones(r_inputs.shape[0])))
    num_inputs = r_inputs.shape[0]

    for i in range(training_steps):

        scores = np.dot(inputs, W1)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        correct_logprobs = -np.log(probs[range(num_inputs), outputs])

        data_loss = np.sum(correct_logprobs) / num_inputs
        reg_loss = 0.5 * reg * (np.sum(W1[:, -1] * W1[:, -1]))
        loss = data_loss + reg_loss
        if i % 100 == 0:
            with open('w1_', 'wb') as fw1:
                pickle._dump(W1, fw1)

            print("iteration %d: data_loss %f, loss %f" % (i, data_loss, loss))
            print(np.sum(scores.argmax(axis=1) == outputs))
            scores = np.maximum(0, np.dot(ev_in, W1))
            print(np.sum(scores.argmax(axis=1) == ev_out))
            sample_images(i, W1, 10050, 0.1)

        dscores = probs
        dscores[range(num_inputs), outputs] -= 1
        dscores /= num_inputs

        dW1 = np.dot(inputs.T, dscores)
        dW1[:, -1] += reg * W1[:, -1]
        W1 += -step_size * dW1

    scores = np.dot(inputs, W1)
    print(np.sum(scores.argmax(axis=1) == outputs))
    scores = np.dot(ev_in, W1)
    print(np.sum(scores.argmax(axis=1) == ev_out))

with open('w1_', 'wb') as fw1:
    pickle._dump(W1, fw1)
