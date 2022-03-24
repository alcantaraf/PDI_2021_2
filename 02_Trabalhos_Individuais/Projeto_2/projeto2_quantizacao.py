import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def quantizacao(img, k):
    L = (2**k) -1
    img_quant = np.float64(img)
    img_quant = img_quant/np.amax(img_quant)
    img_quant = np.uint8(img_quant * L)
    return img_quant


img1 ={}
file_path1 = input("Enter first file path:")
img1[0] = cv.imread(file_path1, 0)

n = [7,6,5,4,3,2,1]
k = [8,7,6,5,4,3,2,1]
aux = 1
for i in n:
    fig, ax = plt.subplots()
    img1[aux] = quantizacao(img1[0],i)
    im1 = ax.imshow(img1[aux], cmap='gray')
    fig.colorbar(im1)
    plt.savefig('Quantizacao/img1qnt_'+str(i) +'.png')
    aux += 1
plt.close('all')

fig1, axes = plt.subplots(2,4, figsize=(12,3))
figManager1 = plt.get_current_fig_manager()
figManager1.window.showMaximized()
aux = 0
for ax in axes.flat:
    im = ax.imshow(img1[aux], cmap = 'gray')
    ax.set_title('k = '+str(k[aux]))
    ax.set_axis_off()
    fig1.colorbar(im, ax=ax)
    aux +=1
plt.tight_layout()
plt.savefig('Quantizacao/img1comp.png')
plt.show()

img2 ={}
file_path2 = input("Enter second file path:")
img2[0] = cv.imread(file_path2, 0)
aux = 1
for i in n:
    fig, ax = plt.subplots()
    img2[aux] =quantizacao(img2[0],i)
    im2 = ax.imshow(img2[aux], cmap='gray')
    fig.colorbar(im2)
    plt.savefig('Quantizacao/img2qnt_'+str(i) +'.png')
    aux +=1
plt.close('all')

fig2, axes = plt.subplots(2,4, figsize=(12,3))
figManager2 = plt.get_current_fig_manager()
figManager2.window.showMaximized()
aux = 0
for ax in axes.flat:
    im = ax.imshow(img2[aux], cmap = 'gray')
    ax.set_title('k = '+str(k[aux]))
    ax.set_axis_off()
    fig2.colorbar(im, ax=ax)
    aux +=1
plt.tight_layout()
plt.savefig('Quantizacao/img2comp.png')
plt.show()

img3 = {}
file_path3 = input("Enter third file path:")
img3[0] = cv.imread(file_path3, 0)
aux = 1
for i in n:
    fig, ax = plt.subplots()
    img3[aux] =quantizacao(img3[0],i)
    im3 = ax.imshow(quantizacao(img3[aux],i), cmap='gray')
    fig.colorbar(im3)
    plt.savefig('Quantizacao/img3qnt_'+str(i) +'.png')   
    aux +=1
plt.close('all')

fig3, axes = plt.subplots(2,4, figsize=(12,3))
figManager3 = plt.get_current_fig_manager()
figManager3.window.showMaximized()
aux = 0
for ax in axes.flat:
    im = ax.imshow(img3[aux], cmap = 'gray')
    ax.set_title('k = '+str(k[aux]))
    ax.set_axis_off()
    fig3.colorbar(im, ax=ax)
    aux +=1
plt.tight_layout()
plt.savefig('Quantizacao/img3comp.png')
plt.show()

#fig, ax = plt.subplots(4,4)
#for i in range(3):
#    for j in range(3):
#        im = ax[i,j].imshow(img1[i+j])
#        plt.colorbar(im,ax=ax[i,j])

#plt.tight_layout()
#plt.show()
#for i in range(1,8):
#    fig, ax = plt.subplots()
#    im = ax.imshow(quantizacao(img2,i), cmap='gray')
#    ax.colormap()
#    fig.show()
#    plt.savefig('Quant/img2'+str(int(512/(2**i)))+'.png')

#for i in range(1,8):
#    fig, ax = plt.subplots()
#    im = ax.imshow(quantizacao(img3,i), cmap='gray')
#    ax.colormap()
#    fig.show()
#    plt.savefig('Quant/img3'+str(int(512/(2**i)))+'.png')