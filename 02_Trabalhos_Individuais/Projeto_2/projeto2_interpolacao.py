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

def upsampleNNI(img, upfactor):
    """This function does the upsample with NNI method for a square input Image"""
    h = img.shape[0]
    w = img.shape[1]
    if not(math.log2(upfactor).is_integer() and h == w):
        print("ERRO: This resize just made power of 2 recizing.")
        
        print("Figure size: "+str(h)+","+str(w))
        print("Reduction: "+str(upfactor))
        sys.exit()
    else:
        upsampled_img = np.kron(img, np.ones((upfactor,upfactor),dtype= np.uint8))
        upsampled_img = np.uint8(upsampled_img)
    return upsampled_img

def upsampleBI(img, upfactor):
    """This fucntion does the bilinear interpolation of the input Image"""
    h = img.shape[0]
    w = img.shape[1]
    if not(math.log2(upfactor).is_integer() and h == w):
        print("This down resize just made power of 2 recizing") 
        return img    
    else: 
        n_dimension = int(img.shape[0]*upfactor)
        new_image = np.zeros((n_dimension,n_dimension), dtype = np.uint8)
        for i in range(n_dimension-1):
            pre_row = int(np.floor(i*h/n_dimension))
            pos_row = int(np.ceil(i*h/n_dimension))
            drow = (i - pre_row)/n_dimension
            if pos_row == h:
                    pos_row = pre_row
            for j in range(n_dimension-1):
                pre_col = int(np.floor(j*w/n_dimension))
                pos_col = int(np.ceil(j*w/n_dimension))
                dcol = (j - pre_col)/n_dimension
                if pos_col == w:
                    pos_col = pre_col
                new_image[i,j] = ((1-drow)*(1-dcol)*img[pre_row,pre_col]) + \
                                    (drow*(1-dcol)*img[pos_row,pre_col]) + \
                                    ((1-drow)*dcol*img[pre_row,pos_col]) + \
                                    (drow*dcol*img[pos_row,pos_col])   
        new_image = np.uint8(new_image)       
    return new_image

def scale(img):
    img = np.uint64(img)
    np.linalg.norm(img)
    return img

def img_diff(original, upsampled):
    original = scale(original)
    upsampled = scale(upsampled)
    difference = np.sqrt((upsampled - original)**2)
    difference = np.uint8(difference * 255)
    return difference

def psnr_comparation(original, cicles, fname):
    nni_psnr = np.zeros(cicles)
    bi_psnr  = np.zeros(cicles)
    iterations = np.zeros(cicles)
    aux = 0 
    fig = plt.figure()
    for i in range(cicles,0,-1):
        I = 2**i 
        downsized_fig = downsampling(original,I)
        nni_upsized = upsampleNNI(downsized_fig, I)
        bi_upsized = upsampleBI(downsized_fig,I)
        nni_psnr[aux] = 10*math.log10((255**2)/(np.mean(original-nni_upsized))**2)
        bi_psnr[aux] = 10*math.log10((255**2)/(np.mean(original-bi_upsized))**2)
        iterations[aux] = I
        aux = aux + 1
    plt.plot(iterations,nni_psnr,'--ro',label='NNI')
    plt.plot(iterations,bi_psnr,'--bo',label='Bilinear')
    plt.grid()
    plt.legend()
    plt.ylabel('PSNR(dB)')
    plt.xlabel('Ratio(n)')
    plt.savefig(fname)


n = [2,4,8,16]

file_path1 = input("Enter first file path:")
img1 = cv.imread(file_path1, 0)

img1_resNNI = {}
img1_resNNI[0] = img1
img1_diffNNI = {}
img1_resBI = {}
img1_diffBI = {}
img1_resBI[0] = img1
aux = 1

for i in n:
    img1Down = downsampling(img1, i)
    img1_resNNI[aux] = upsampleNNI(img1Down, i)

    img1_diffNNI[aux-1] = img_diff(img1,img1_resNNI[aux])
    img1_resBI[aux] = upsampleBI(img1Down, i)
    img1_diffBI[aux-1] = img_diff(img1,img1_resBI[aux])
    aux += 1

for i in range(aux):
    fig, ax = plt.subplots()
    ax.imshow(img1_resNNI[i], cmap='gray')
    plt.savefig('Up/img1NNI_'+ str(int(512/(2**i)))+'.png')
    fig, ax = plt.subplots()
    ax.imshow(img1_resBI[i], cmap='gray')
    plt.savefig('Up/img1BI_'+ str(int(512/(2**i)))+'.png')

for i in img1_diffNNI:
    fig, ax = plt.subplots()
    ax.imshow(img1_diffNNI[i], cmap='gray')
    plt.savefig('Up/img1NNIdiff_'+ str(int(256/(2**i)))+'.png')
    fig, ax = plt.subplots()
    ax.imshow(img1_diffBI[i], cmap='gray')
    plt.savefig('Up/img1BIdiff_'+ str(int(256/(2**i)))+'.png')

psnr_comparation(img1, aux-1, 'Up/img1PSNR')
plt.close('all')
file_path2 = input("Enter second file path:")
img2 = cv.imread(file_path2, 0)

img2_resNNI = {}
img2_resNNI[0] = img2
img2_diffNNI = {}
img2_resBI = {}
img2_diffBI = {}
img2_resBI[0] = img2
aux = 1

for i in n:
    img2Down = downsampling(img2, i)
    img2_resNNI[aux] = upsampleNNI(img2Down, i)

    img2_diffNNI[aux-1] = img_diff(img2,img2_resNNI[aux])
    img2_resBI[aux] = upsampleBI(img2Down, i)
    img2_diffBI[aux-1] = img_diff(img2,img2_resBI[aux])
    aux += 1

for i in range(aux):
    fig, ax = plt.subplots()
    ax.imshow(img2_resNNI[i], cmap='gray')
    plt.savefig('Up/img2NNI_'+ str(int(512/(2**i)))+'.png')
    fig, ax = plt.subplots()
    ax.imshow(img2_resBI[i], cmap='gray')
    plt.savefig('Up/img2BI_'+ str(int(512/(2**i)))+'.png')

for i in img2_diffNNI:
    fig, ax = plt.subplots()
    ax.imshow(img2_diffNNI[i], cmap='gray')
    plt.savefig('Up/img2NNIdiff_'+ str(int(256/(2**i)))+'.png')
    fig, ax = plt.subplots()
    ax.imshow(img2_diffBI[i], cmap='gray')
    plt.savefig('Up/img2BIdiff_'+ str(int(256/(2**i)))+'.png')

psnr_comparation(img2, aux-1, 'Up/img2PSNR')

plt.close('all')

fig1 = plt.figure(figsize=(24,5))
ax1_1 = plt.subplot(2,4,1)
ax1_2 = plt.subplot(2,4,2)
ax1_3 = plt.subplot(2,4,3)
ax1_4 = plt.subplot(2,4,4)
ax1_5 = plt.subplot(2,4,5)
ax1_7 = plt.subplot(2,4,6)
ax1_8 = plt.subplot(2,4,7)
ax1_9 = plt.subplot(2,4,8)
ax1_1.imshow(img1_diffNNI[0],cmap='gray', vmin = 0, vmax = 255)
ax1_1.set_title('NNI - 256x256')
ax1_1.set_axis_off()
ax1_2.imshow(img1_diffNNI[1],cmap='gray', vmin = 0, vmax = 255)
ax1_2.set_title('NNI - 128x128')
ax1_2.set_axis_off()
ax1_3.imshow(img1_diffNNI[2],cmap='gray', vmin = 0, vmax = 255)
ax1_3.set_title('NNI - 64x64')
ax1_3.set_axis_off()
ax1_4.imshow(img1_diffNNI[3],cmap='gray', vmin = 0, vmax = 255)
ax1_4.set_title('NNI - 32x32')
ax1_4.set_axis_off()
ax1_5.imshow(img1_diffBI[0],cmap='gray', vmin = 0, vmax = 255)
ax1_5.set_title('Bi - 256x256')
ax1_5.set_axis_off()
ax1_7.imshow(img1_diffBI[1],cmap='gray', vmin = 0, vmax = 255)
ax1_7.set_title('Bi - 128x128')
ax1_7.set_axis_off()
ax1_8.imshow(img1_diffBI[2],cmap='gray', vmin = 0, vmax = 255)
ax1_8.set_title('Bi - 64x64')
ax1_8.set_axis_off()
ax1_9.imshow(img1_diffBI[3],cmap='gray', vmin = 0, vmax = 255)
ax1_9.set_title('Bi - 32x32')
ax1_9.set_axis_off()
plt.tight_layout()
plt.savefig('Up/img1comp.png')
plt.show()

fig1 = plt.figure(figsize=(24,5))
ax1_1 = plt.subplot(2,4,1)
ax1_2 = plt.subplot(2,4,2)
ax1_3 = plt.subplot(2,4,3)
ax1_4 = plt.subplot(2,4,4)
ax1_5 = plt.subplot(2,4,5)
ax1_7 = plt.subplot(2,4,6)
ax1_8 = plt.subplot(2,4,7)
ax1_9 = plt.subplot(2,4,8)
ax1_1.imshow(img2_diffNNI[0],cmap='gray', vmin = 0, vmax = 255)
ax1_1.set_title('NNI - 256x256')
ax1_1.set_axis_off()
ax1_2.imshow(img2_diffNNI[1],cmap='gray', vmin = 0, vmax = 255)
ax1_2.set_title('NNI - 128x128')
ax1_2.set_axis_off()
ax1_3.imshow(img2_diffNNI[2],cmap='gray', vmin = 0, vmax = 255)
ax1_3.set_title('NNI - 64x64')
ax1_3.set_axis_off()
ax1_4.imshow(img2_diffNNI[3],cmap='gray', vmin = 0, vmax = 255)
ax1_4.set_title('NNI - 32x32')
ax1_4.set_axis_off()
ax1_5.imshow(img2_diffBI[0],cmap='gray', vmin = 0, vmax = 255)
ax1_5.set_title('Bi - 256x256')
ax1_5.set_axis_off()
ax1_7.imshow(img2_diffBI[1],cmap='gray', vmin = 0, vmax = 255)
ax1_7.set_title('Bi - 128x128')
ax1_7.set_axis_off()
ax1_8.imshow(img2_diffBI[2],cmap='gray', vmin = 0, vmax = 255)
ax1_8.set_title('Bi - 64x64')
ax1_8.set_axis_off()
ax1_9.imshow(img2_diffBI[3],cmap='gray', vmin = 0, vmax = 255)
ax1_9.set_title('Bi - 32x32')
ax1_9.set_axis_off()
plt.tight_layout()
plt.savefig('Up/img2comp.png')
plt.show()

'''
image1_d256 = downsampling(img1, 2)
image1_u256NNI = upsampleNNI(image1_d256,2)
image1_u256BI = upsampleBI(image1_d256,2)
fig = plt.figure()
plt.imshow(image1_u256NNI, cmap='gray')
#plt.show()
#fname = input("NNI 256:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow(image1_u256BI, cmap='gray')
#plt.show()
#fname = input("BI 256:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow((img1-image1_u256NNI), cmap='gray')
#plt.show()
#fname = input("ORIGNAL - NNI:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow((img1-image1_u256BI), cmap='gray')
#plt.show()
#fname = input("ORIGNAL - BI:")
#fig.savefig(fname + '.png')

image1_d128 = downsampling(img1, 4)
image1_u128NNI = upsampleNNI(image1_d128,4)
image1_u128BI = upsampleBI(image1_d128,4)
fig = plt.figure()
plt.imshow(image1_u128NNI, cmap='gray')
#plt.show()
#fname = input("NNI 128:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow(image1_u128BI, cmap='gray')
#plt.show()
#fname = input("BI 128:")
#fig.savefig(fname + '.png')
fig = plt.figure()
#plt.imshow((img1-image1_u128NNI), cmap='gray')
#plt.show()
#fname = input("ORIGNAL - NNI:")
#fig.savefig(fname + '.png')
fig = plt.figure()
#plt.imshow((img1-image1_u128BI), cmap='gray')
#plt.show()
#fname = input("ORIGNAL - BI:")
#fig.savefig(fname + '.png')


image1_d64  = downsampling(img1, 8)
image1_u64NNI = upsampleNNI(image1_d64,8)
image1_u64BI = upsampleBI(image1_d64,8)
fig = plt.figure()
plt.imshow(image1_u64NNI, cmap='gray')
#plt.show()
#fname = input("NNI 64:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow(image1_u64BI, cmap='gray')
#plt.show()
#fname = input("BI 64:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow((img1-image1_u64NNI), cmap='gray')
#plt.show()
#fname = input("ORIGNAL - NNI:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow((img1-image1_u64BI), cmap='gray')
#plt.show()
#fname = input("ORIGNAL - BI:")
#fig.savefig(fname + '.png')


image1_d32  = downsampling(img1, 16)
image1_u32NNI = upsampleNNI(image1_d32,16)
image1_u32BI = upsampleBI(image1_d32,16)
fig = plt.figure()
plt.imshow(image1_u32NNI, cmap='gray')
#plt.show()
#fname = input("NNI 32:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow(image1_u32BI, cmap='gray')
#plt.show()
#fname = input("BI 32:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow((img1-image1_u32NNI), cmap='gray')
#plt.show()
#fname = input("ORIGNAL - NNI:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow((img1-image1_u32BI), cmap='gray')
#plt.show()
#fname = input("ORIGNAL - BI:")
#fig.savefig(fname + '.png')


fig1 = plt.figure(figsize=(24,5))
ax1_1 = plt.subplot(1,5,1)
ax1_2 = plt.subplot(1,5,2)
ax1_3 = plt.subplot(1,5,3)
ax1_4 = plt.subplot(1,5,4)
ax1_5 = plt.subplot(1,5,5)
ax1_1.imshow(img1,cmap='gray')
ax1_1.set_title('Original')
ax1_2.imshow(image1_u256NNI,cmap='gray')
ax1_2.set_title('256x256')
ax1_3.imshow(image1_u128NNI,cmap='gray')
ax1_3.set_title('128x128')
ax1_4.imshow(image1_u64NNI,cmap='gray')
ax1_4.set_title('64x64')
ax1_5.imshow(image1_u32NNI,cmap='gray')
ax1_5.set_title('32x32')
plt.tight_layout()
#fname = input("Nome para a Figura comparativa interpolacao de vizinhos proximos 1:")
#fig1.savefig(fname + '.png')
#plt.show()

fig1 = plt.figure(figsize=(24,5))
ax1_1 = plt.subplot(1,5,1)
ax1_2 = plt.subplot(1,5,2)
ax1_3 = plt.subplot(1,5,3)
ax1_4 = plt.subplot(1,5,4)
ax1_5 = plt.subplot(1,5,5)
ax1_1.imshow(img1,cmap='gray')
ax1_1.set_title('Original')
ax1_2.imshow(image1_u256BI,cmap='gray')
ax1_2.set_title('256x256')
ax1_3.imshow(image1_u128BI,cmap='gray')
ax1_3.set_title('128x128')
ax1_4.imshow(image1_u64BI,cmap='gray')
ax1_4.set_title('64x64')
ax1_5.imshow(image1_u32BI,cmap='gray')
ax1_5.set_title('32x32')
plt.tight_layout()
#fname = input("Nome para a Figura comparativa interpolacao bilinear 1:")
#fig1.savefig(fname + '.png')
#plt.show()


file_path2 = input("Enter second file path:")
img2 = cv.imread(file_path2, 0)
image2_d256 = downsampling(img2, 2)
image2_u256NNI = upsampleNNI(image2_d256, 2)
image2_u256BI = upsampleBI(image2_d256, 2)
fig = plt.figure()
plt.imshow(image2_u256NNI, cmap='gray')
#plt.show()
#fname = input("NNI 256:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow(image2_u256BI, cmap='gray')
#plt.show()
#fname = input("BI 256:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow((img2-image2_u256NNI), cmap='gray')
#plt.show()
#fname = input("ORIGNAL - NNI:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow((img2-image2_u256BI), cmap='gray')
#plt.show()
#fname = input("ORIGNAL - BI:")
#fig.savefig(fname + '.png')


image2_d128 = downsampling(img2, 4)
image2_u128NNI = upsampleNNI(image2_d128, 4)
image2_u128BI = upsampleBI(image2_d128, 4)
fig = plt.figure()
plt.imshow(image2_u128NNI, cmap='gray')
#plt.show()
#fname = input("NNI 128:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow(image2_u128BI, cmap='gray')
#plt.show()
#fname = input("BI 128:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow((img2-image2_u128NNI), cmap='gray')
#plt.show()
#fname = input("ORIGNAL - NNI:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow((img2-image2_u128BI), cmap='gray')
#plt.show()
#fname = input("ORIGNAL - BI:")
#fig.savefig(fname + '.png')


image2_d64  = downsampling(img2, 8)
image2_u64NNI = upsampleNNI(image2_d64, 8)
image2_u64BI = upsampleBI(image2_d64, 8)
fig = plt.figure()
plt.imshow(image2_u64NNI, cmap='gray')
#plt.show()
#fname = input("NNI 64:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow(image2_u64BI, cmap='gray')
#plt.show()
#fname = input("BI 64:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow((img2-image2_u64NNI), cmap='gray')
#plt.show()
#fname = input("ORIGNAL - NNI:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow((img2-image2_u64BI), cmap='gray')
#plt.show()
#fname = input("ORIGNAL - BI:")
#fig.savefig(fname + '.png')


image2_d32  = downsampling(img2, 16)
image2_u32NNI = upsampleNNI(image2_d32, 16)
image2_u32BI = upsampleBI(image2_d32, 16)
fig = plt.figure()
plt.imshow(image2_u32NNI, cmap='gray')
#plt.show()
#fname = input("NNI 32:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow(image2_u32BI, cmap='gray')
#plt.show()
#fname = input("BI 32:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow((img2-image2_u32NNI), cmap='gray')
#plt.show()
#fname = input("ORIGNAL - NNI:")
#fig.savefig(fname + '.png')
fig = plt.figure()
plt.imshow((img2-image2_u32BI), cmap='gray')
#plt.show()
#fname = input("ORIGNAL - BI:")
#fig.savefig(fname + '.png')


fig2 = plt.figure(figsize=(24,5))
ax2_1 = plt.subplot(1,5,1)
ax2_2 = plt.subplot(1,5,2)
ax2_3 = plt.subplot(1,5,3)
ax2_4 = plt.subplot(1,5,4)
ax2_5 = plt.subplot(1,5,5)
ax2_1.imshow(img2,cmap='gray')
ax2_1.set_title('Original')
ax2_2.imshow(image2_u256NNI,cmap='gray')
ax2_2.set_title('256x256')
ax2_3.imshow(image2_u128NNI,cmap='gray')
ax2_3.set_title('128x128')
ax2_4.imshow(image2_u64NNI,cmap='gray')
ax2_4.set_title('64x64')
ax2_5.imshow(image2_u32NNI,cmap='gray')
ax2_5.set_title('32x32')
plt.tight_layout()
#fname = input("Nome para a Figura comparativa interpolacao de vizinhos proximos 2:")
#fig2.savefig(fname + '.png')
#plt.show()

fig2 = plt.figure(figsize=(24,5))
ax2_1 = plt.subplot(1,5,1)
ax2_2 = plt.subplot(1,5,2)
ax2_3 = plt.subplot(1,5,3)
ax2_4 = plt.subplot(1,5,4)
ax2_5 = plt.subplot(1,5,5)
ax2_1.imshow(img2,cmap='gray')
ax2_1.set_title('Original')
ax2_2.imshow(image2_u256BI,cmap='gray')
ax2_2.set_title('256x256')
ax2_3.imshow(image2_u128BI,cmap='gray')
ax2_3.set_title('128x128')
ax2_4.imshow(image2_u64BI,cmap='gray')
ax2_4.set_title('64x64')
ax2_5.imshow(image2_u32BI,cmap='gray')
ax2_5.set_title('32x32')
plt.tight_layout()
#fname = input("Nome para a Figura comparativa interpolacao bilinear 2 :")
#fig2.savefig(fname + '.png')
#plt.show()
plt.show()
'''