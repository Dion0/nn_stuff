import numpy as np
import matplotlib.pyplot as plt

Z = np.random.rand(128, 255)


plt.ion()

size = np.array(Z.shape)
dpi = 72.0
figsize= size[1]/float(dpi),size[0]/float(dpi)
fig = plt.figure(figsize=figsize, dpi=int(dpi), facecolor="white")
fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)

im = plt.imshow(Z, interpolation='nearest', cmap=plt.cm.gray_r)

im.set_data(Z)
plt.draw()
plt.pause(0.01)

Z = np.cos(Z)
im.set_data(Z)
plt.draw()

plt.savefig("./tmp-%03d.png" % 10,dpi=dpi)
plt.pause(2)

plt.ioff()