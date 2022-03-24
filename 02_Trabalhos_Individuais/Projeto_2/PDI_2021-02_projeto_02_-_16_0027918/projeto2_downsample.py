import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import sys


def downsampling(img, downfactor):
    """This function does the downsampling of a square input Image"""
    if not(math.log2(downfactor).is_integer() and img.shape[0] == img.shape[1]):
        print("ERRO: This resize just made power of 2 recizing.")
        h = img.shape[0]
        w = img.shape[1]
        print("Figure size: "+str(h)+","+str(w))
        print("Down Factor: "+str(downfactor))
        sys.exit()
    else:
        n_dimension = int(img.shape[0]/downfactor)
        new_image = np.zeros((n_dimension,n_dimension), dtype= np.uint8)
        auxj = 0
        auxi = 0
        j_count = -1
        i_count = -1
        for i in range(img.shape[0]-1):
            i_count = i_count +1
            if i_count == downfactor:
                i_count = 0
            if i_count == 0:
                for j in range(img.shape[1]-1):
                    j_count = j_count +1
                    if j_count == downfactor:
                        j_count = 0 
                    if j_count == 0:
                        new_image[auxi,auxj] = img[i,j]
                        auxj = auxj + 1
                auxj = 0
                j_count = -1
                auxi = auxi +1
        new_image = np.uint8(new_image)
    return new_image                                 

file_path1 = input("Enter first file path:")
img1 = cv.imread(file_path1, 0)

file_path2 = input("Enter second file path:")
img2 = cv.imread(file_path2, 0)


image1_256 = downsampling(img1, 2)
fig = plt.figure()
plt.imshow(image1_256, cmap='gray')
fname = input("Nome para o downsample (d = 2) da Imagem 1:")
fig.savefig(fname + '.png')
image1_128 = downsampling(img1, 4)
fig = plt.figure()
plt.imshow(image1_128, cmap='gray')
fname = input("Nome para o downsample (d = 4) da Imagem 1:")
fig.savefig(fname + '.png')
image1_64  = downsampling(img1, 8)
fig = plt.figure()
plt.imshow(image1_64, cmap='gray')
fname = input("Nome para o downsample (d = 8) da Imagem 1:")
fig.savefig(fname + '.png')
image1_32  = downsampling(img1, 16)
fig = plt.figure()
plt.imshow(image1_32, cmap='gray')
fname = input("Nome para o downsample (d = 16) da Imagem 1:")
fig.savefig(fname + '.png')

fig1 = plt.figure(figsize=(24,5))
ax1_1 = plt.subplot(1,5,1)
ax1_2 = plt.subplot(1,5,2)
ax1_3 = plt.subplot(1,5,3)
ax1_4 = plt.subplot(1,5,4)
ax1_5 = plt.subplot(1,5,5)
ax1_1.imshow(img1,cmap='gray', vmin = 0, vmax = 255)
ax1_1.set_title('Original')
ax1_2.imshow(image1_256,cmap='gray', vmin = 0, vmax = 255)
ax1_2.set_title('256x256')
ax1_3.imshow(image1_128,cmap='gray', vmin = 0, vmax = 255)
ax1_3.set_title('128x128')
ax1_4.imshow(image1_64,cmap='gray', vmin = 0, vmax = 255)
ax1_4.set_title('64x64')
ax1_5.imshow(image1_32,cmap='gray', vmin = 0, vmax = 255)
ax1_5.set_title('32x32')
plt.tight_layout()
fname = input("Nome para a Figura comparativa 1:")
fig1.savefig(fname + '.png')
plt.show()

image2_256 = downsampling(img2, 2)
fig = plt.figure()
plt.imshow(image2_256, cmap='gray')
fname = input("Nome para o downsample (d = 2) da Imagem 2:")
fig.savefig(fname + '.png')
image2_128 = downsampling(img2, 4)
fig = plt.figure()
plt.imshow(image2_128, cmap='gray')
fname = input("Nome para o downsample (d = 4) da Imagem 2:")
fig.savefig(fname + '.png')
image2_64  = downsampling(img2, 8)
fig = plt.figure()
plt.imshow(image2_64, cmap='gray')
fname = input("Nome para o downsample (d = 8) da Imagem 2:")
fig.savefig(fname + '.png')
image2_32  = downsampling(img2, 16)
fig = plt.figure()
plt.imshow(image2_32, cmap='gray')
fname = input("Nome para o downsample (d = 16) da Imagem 2:")
fig.savefig(fname + '.png')

fig2 = plt.figure(figsize=(24,5))
ax2_1 = plt.subplot(1,5,1)
ax2_2 = plt.subplot(1,5,2)
ax2_3 = plt.subplot(1,5,3)
ax2_4 = plt.subplot(1,5,4)
ax2_5 = plt.subplot(1,5,5)
ax2_1.imshow(img2,cmap='gray', vmin = 0, vmax = 255)
ax2_1.set_title('Original')
ax2_2.imshow(image2_256,cmap='gray', vmin = 0, vmax = 255)
ax2_2.set_title('256x256')
ax2_3.imshow(image2_128,cmap='gray', vmin = 0, vmax = 255)
ax2_3.set_title('128x128')
ax2_4.imshow(image2_64,cmap='gray', vmin = 0, vmax = 255)
ax2_4.set_title('64x64')
ax2_5.imshow(image2_32,cmap='gray', vmin = 0, vmax = 255)
ax2_5.set_title('32x32')
plt.tight_layout()
fname = input("Nome para a Figura comparativa 2:")
fig2.savefig(fname + '.png')
plt.show()


