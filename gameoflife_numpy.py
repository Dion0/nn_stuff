import numpy as np
import matplotlib.pyplot as plt

Z = np.random.randint(0, 2, (256, 512))

def iterate(Z):
    assert type(Z) is np.ndarray

    A = (Z[:-2, :-2] + Z[1:-1, :-2] + Z[2:, :-2]
         + Z[:-2, 1:-1] + Z[2:, 1:-1]
         + Z[:-2, 2:] + Z[1:-1, 2:] + Z[2:, 2:])

    birth = (A == 3) & (Z[1:-1,1:-1] == 0)
    survive = ((A == 2) | (A == 3)) & (Z[1:-1, 1:-1] == 1)
    Z[...] = 0
    Z[1:-1, 1:-1][birth | survive] = 1

    return Z

def show(Z):
    assert type(Z) is np.ndarray

    size = np.array(Z.shape)
    dpi = 72.0
    figsize = size[1] / dpi, size[0] / dpi
    fig = plt.figure(figsize=figsize, dpi=int(dpi), facecolor='white')
    fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon = False)
    plt.imshow(Z, interpolation='nearest', cmap = plt.cm.gray_r)
    plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':

    for i in range(100):
        iterate(Z)
    show(Z)