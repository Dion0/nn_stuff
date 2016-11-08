import pickle
from scipy.misc import imread, imsave, imresize
import numpy as np


if __name__ == "__main__":
    img = imread('in.png')
    img = imresize(img, (300, 300))
    imsave('out.png', img)

    key = 45

    img_r = np.ravel(img)
    used_r = np.zeros(img_r.shape)
    ind = key
    cur_key = key
    ar_len = img_r.shape[0]
    for i in range(ar_len):
        while used_r[ind] == 1:
            ind = (ind + 1) % ar_len
        used_r[ind] = 1
        cur_key = img_r[ind] = (img_r[ind] + cur_key) % 256
        ind = (ind + cur_key) % ar_len

    imsave('out_encoded.png', img)
    print('encoding dun')

    used_r = np.zeros(img_r.shape)
    ind = key
    cur_key = key
    next_ind = 0
    print(img_r)
    for i in range(ar_len):
        while used_r[ind] == 1:
            ind = ind + 1
            if ind >= ar_len:
                ind -= ar_len
        next_ind = img_r[ind]
        used_r[ind] = 1
        if img_r[ind] < cur_key:
            tmp = cur_key - img_r[ind]
            img_r[ind] = 255 - tmp + 1
        else:
            img_r[ind] -= cur_key
        if img_r[ind] < 0:
            img_r[ind] += 256
        cur_key = next_ind
        ind += next_ind
        if ind >= ar_len:
            ind -= ar_len

    imsave('out_decoded.png', img)
