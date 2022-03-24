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

def quantizacao(img, k):
    L = (2**k) -1
    img_quant = np.float64(img)
    img_quant = img_quant/np.amax(img_quant)
    img_quant = np.uint8(img_quant * L)
    return img_quant
k = [6,5,4,3]
n = [2,4,8,16]

img1 ={}
img1_ = cv.imread('/home/felipe/classes2021_2/PDI_2021_2/02_Trabalhos_Individuais/Recursos/woman_blonde.tif', 0)
aux = 0
for i in n:
    for j in k:
        image_aux = downsampling(img1_, i)
        img1[aux] = quantizacao(image_aux,j)
        aux += 1

fig1, axes = plt.subplots(4,4, figsize=(8,8))
figManager1 = plt.get_current_fig_manager()
figManager1.window.showMaximized()
aux = 0
aux1 = 0
for ax in axes.flat:
    im = ax.imshow(img1[aux], cmap = 'gray')
    if(aux1 == 4):
        aux1 = 0
    ax.set_title('k = '+str(k[aux1]))
    fig1.colorbar(im, ax=ax)
    aux +=1
    aux1 +=1
plt.tight_layout()
plt.savefig('simu/img1comp.png')
plt.show()
plt.close('all')
img2 = {}
img2_ = cv.imread('/home/felipe/classes2021_2/PDI_2021_2/02_Trabalhos_Individuais/Recursos/cameraman.tif', 0)

aux = 0
for i in n:
    for j in k:
        image_aux = downsampling(img2_, i)
        img2[aux] = quantizacao(image_aux,j)
        aux += 1

fig2, axes = plt.subplots(4,4, figsize=(8,8))
figManager2 = plt.get_current_fig_manager()
figManager2.window.showMaximized()
aux = 0
aux1 = 0
for ax in axes.flat:
    im = ax.imshow(img2[aux], cmap = 'gray')
    if(aux1 == 4):
        aux1 = 0
    ax.set_title('k = '+str(k[aux1]))
    fig2.colorbar(im, ax=ax)
    aux +=1
    aux1 +=1
plt.tight_layout()
plt.savefig('simu/img2comp.png')
plt.show()
plt.close('all')

img3 = {}
img3_ = cv.imread('/home/felipe/classes2021_2/PDI_2021_2/02_Trabalhos_Individuais/Recursos/walkbridge.tif', 0)

aux = 0
for i in n:
    for j in k:
        image_aux = downsampling(img3_, i)
        img3[aux] = quantizacao(image_aux,j)
        aux += 1

fig3, axes = plt.subplots(4,4, figsize=(8,8))
figManager3 = plt.get_current_fig_manager()
figManager3.window.showMaximized()
aux = 0
aux1 = 0
for ax in axes.flat:
    im = ax.imshow(img3[aux], cmap = 'gray')
    if(aux1 == 4):
        aux1 = 0
    ax.set_title('k = '+str(k[aux1]))
    fig3.colorbar(im, ax=ax)
    aux +=1
    aux1 +=1
plt.tight_layout()
plt.savefig('simu/img3comp.png')
plt.show()
plt.close('all')