from scipy.misc import imread, imsave, imresize

img = imread("goomi.png")
print(img.dtype, img.shape)

img_tinted = img * [0.7, 0.6, 1.0, 1.0]
img_tinted[:,:,2] = 255

img_tinted = imresize(img_tinted, (300, 300))


imsave("goomier.png", img_tinted)

