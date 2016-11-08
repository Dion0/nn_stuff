import pickle
from scipy.misc import imread, imsave, imresize
import numpy as np


PREFIX = 'cifar\\cifar-10-batches-py\\'
IMG_PREFIX = 'cifar_out\\'

decode_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def unpickle(file):
    full_name = PREFIX + file
    with open(full_name, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        dict = u.load()
    return dict

def save_img(arr, resname = 'wow.png'):
    r = arr[:1024]
    g = arr[1024:2048]
    b = arr[2048:3072]
    print(resname, ' ', np.sum(r), ' ', np.sum(g), ' ', np.sum(b))
    rarr = np.array([r, g, b]).transpose()
    imsave(IMG_PREFIX + resname, rarr.reshape(32, 32, 3))

def load_img(resname = 'wow.png'):
    inp = imread(resname, mode='RGB')
    inp = imresize(inp, (32, 32))
    imsave('32_' + resname, inp)
    return inp

if __name__ == "__main__":
    load_img('in.png')
    exit(0)
