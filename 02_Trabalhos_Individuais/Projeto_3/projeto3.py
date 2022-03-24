import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import seaborn as sns


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

def viz4(img, row, col, label):
    up = img[row-1][col]
    left = img[row][col-1]

    if (up in label) and (up != img[row][col]):
        return up
    elif (left in label) and (left != img[row][col]):
        return left
    else:
        return 0

def firstlabel(img):
    label=[]
    i = 1
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if((img[row][col] != 0)):
                flag = viz4(img, row, col, label)
                if flag != 0:
                    img[row][col] = flag  
                else:
                    img[row][col] = i
                    label.append(i)
                    i += 1  
                    
    return img, label

def secondlabel(img, label):
    for row in range(img.shape[0]-1):
        for col in range(img.shape[1]-1):
            pixel = img[row][col]
            if((pixel in label)):
                flag = viz4(img, row, col, label)
                if flag != 0:
                    if pixel < flag:
                        label.remove(flag)
                        img[img == flag] = pixel
                    else:
                        label.remove(pixel)
                        img[img == pixel] = flag
    for i, val in enumerate(label, 1):
        img[img == val] = i
    label = np.arange(1, len(label)+1 )
    return img, label

def classAB(img, label):
    A,B = [],[]
    buracos = 0
    for i in label:
        #plt.figure(i)
        img_ = np.copy(img)
        #plt.subplot(221)
        #sns.heatmap(img, annot=True, cmap=sns.color_palette("rocket", as_cmap=True),linewidths=0.5)
        img_[img_ != i] = 0
        img_[img_ == i] = 1 
        img_ = quantizacao(cv.bitwise_not(img_),1)
        #plt.subplot(222)
        #sns.heatmap(img_, annot=True,cmap=sns.color_palette("mako", as_cmap=True),linewidths=0.5)
        img_, label_ = firstlabel(img_)
        #plt.subplot(223)
        #sns.heatmap(img_, annot=True, cmap= sns.color_palette("viridis", as_cmap=True),linewidths=0.5)
        img_, label_ = secondlabel(img_, label_)
        #plt.subplot(224)
        #sns.heatmap(img_, annot=True, cmap= sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),linewidths=0.5)
        if(len(label_)) > 1:
            B.append(i)
            buracos = buracos + (len(label_) - 1)
        else:
            A.append(i)
        #plt.tight_layout()
        #plt.show()
    #plt.close('all')
    return A, B, buracos



img1 = cv.imread("fig1.tif",0)

img1_ = quantizacao(cv.bitwise_not(img1),1)
plt.figure()
ax = plt.subplot(131)
plt.imshow(img1_,cmap='gray')
img1_fl, label1_fl = firstlabel(img1_)
plt.colorbar(ax = ax,shrink = .39)
ax = plt.subplot(132)
plt.imshow(img1_fl, cmap= sns.color_palette("icefire", as_cmap=True))
plt.colorbar(ax = ax,shrink = .39)
#sns.heatmap(img1_fl, annot=True, cmap= sns.color_palette("coolwarm", as_cmap=True),linewidths=.8, center = 1)
img1_sl, label1_sl = secondlabel(img1_fl, label1_fl)
ax = plt.subplot(133)
plt.imshow(img1_sl, cmap='gray')
plt.colorbar(ax = ax,shrink = .39)
plt.tight_layout()


A1,B1, buracos1 = classAB(img1_sl, label1_sl)
print('Imagem 1')
print('Tipo A: '+str(len(A1))+'\nTipo B: ' +str(len(B1))+'\nBuracos: '+str(buracos1))

img2 = cv.imread("fig2.tif",0)
img2_ = quantizacao(cv.bitwise_not(img2),1)
plt.figure()
ax = plt.subplot(131)
plt.imshow(img2_,cmap='gray')
img2_fl, label2_fl = firstlabel(img2_)
plt.colorbar(ax = ax,shrink = .39)
ax = plt.subplot(132)
plt.imshow(img2_fl, cmap= sns.color_palette("icefire", as_cmap=True))
plt.colorbar(ax = ax,shrink = .39)
#sns.heatmap(img2_fl, annot=True, cmap= sns.color_palette("coolwarm", as_cmap=True),linewidths=.8, center = 1)
img2_sl, label2_sl = secondlabel(img2_fl, label2_fl)
ax = plt.subplot(133)
plt.imshow(img2_sl, cmap='gray')
plt.colorbar(ax = ax,shrink = .39)


A2,B2, buracos2 = classAB(img2_sl, label2_sl)
print('Imagem 2')
print('Tipo A: '+str(len(A2))+'\nTipo B: ' +str(len(B2))+'\nBuracos: '+str(buracos2))


img3 = cv.imread("fig3.tif",0)
img3_ = quantizacao(cv.bitwise_not(img3),1)
plt.figure()
ax = plt.subplot(131)
plt.imshow(img3_,cmap='gray')
img3_fl, label3_fl = firstlabel(img3_)
plt.colorbar(ax = ax,shrink = .39)
ax = plt.subplot(132)
plt.imshow(img3_fl, cmap= sns.color_palette("icefire", as_cmap=True))
plt.colorbar(ax = ax,shrink = .39)
#sns.heatmap(img3_fl, annot=True, cmap= sns.color_palette("coolwarm", as_cmap=True),linewidths=.8, center = 1)
img3_sl, label3_sl = secondlabel(img3_fl, label3_fl)
ax = plt.subplot(133)
plt.imshow(img3_sl, cmap='gray')
plt.colorbar(ax = ax,shrink = .39)

A3,B3, buracos3 = classAB(img3_sl, label3_sl)
print('Imagem 3')
print('Tipo A: '+str(len(A3))+'\nTipo B: ' +str(len(B3))+'\nBuracos: '+str(buracos3))


img4 = cv.imread("fig4.tif",0)
img4_ = quantizacao(cv.bitwise_not(img4),1)
plt.figure()
ax = plt.subplot(131)
plt.imshow(img4_,cmap='gray')
img4_fl, label4_fl = firstlabel(img4_)
plt.colorbar(ax = ax,shrink = .39)
ax = plt.subplot(132)
plt.imshow(img4_fl, cmap= sns.color_palette("icefire", as_cmap=True))
plt.colorbar(ax = ax,shrink = .39)
#sns.heatmap(img4_fl, annot=True, cmap= sns.color_palette("coolwarm", as_cmap=True),linewidths=.8, center = 1)
img4_sl, label4_sl = secondlabel(img4_fl, label4_fl)
ax = plt.subplot(133)
plt.imshow(img4_sl, cmap=sns.color_palette("icefire", as_cmap=True))
plt.colorbar(ax = ax,shrink = .39)


A4,B4, buracos4 = classAB(img4_sl, label4_sl)
print('Imagem 4')
print('Tipo A: '+str(len(A4))+'\nTipo B: ' +str(len(B4))+'\nBuracos: '+str(buracos4))


img5 = cv.imread("fig5.tif",0)
img5_ = quantizacao(cv.bitwise_not(img5),1)
plt.figure()
ax = plt.subplot(131)
plt.imshow(img5_,cmap='gray')
img5_fl, label5_fl = firstlabel(img5_)
plt.colorbar(ax = ax,shrink = .39)
ax = plt.subplot(132)
plt.imshow(img5_fl, cmap= sns.color_palette("icefire", as_cmap=True))
plt.colorbar(ax = ax,shrink = .39)
#sns.heatmap(img5_fl, annot=True, cmap= sns.color_palette("coolwarm", as_cmap=True),linewidths=.8, center = 1)
img5_sl, label5_sl = secondlabel(img5_fl, label5_fl)
ax = plt.subplot(133)
plt.imshow(img5_sl, cmap=sns.color_palette("icefire", as_cmap=True))
plt.colorbar(ax = ax,shrink = .39)



A5,B5, buracos5 = classAB(img5_sl, label5_sl)
print('Imagem 5')
print('Tipo A: '+str(len(A5))+'\nTipo B: ' +str(len(B5))+'\nBuracos: '+str(buracos5))

plt.show()
plt.close('all')