#!/bin/env python3
#*-* coding=utf-8 *-*

import os,sys
import argparse
import glob
import numpy as np
from skimage import io, transform



#resize picture to smaller size
def resizepic(large_img, args):
    new_shape = (args.newsize, args.newsize, 3)
    small_img = transform.resize(large_img, new_shape)

    return small_img

#minus avg of each channel
def minusavg(ori_img):
    img_shape = ori_img.shape
    img_chflat = np.reshape(ori_img, (img_shape[0]*img_shape[1], img_shape[2]))
    chn_avg = np.mean(img_chflat, axis=1)
    minus_maplist =[]
    chs = ["r","g","b"]
    for ich, ch in enumerate(chs):
        chmap = chn_avg[ich]*np.ones((img_shape[0], img_shape[1]))
        minus_maplist.append(chmap)
    minus_map = np.stack((minus_maplist[0],minus_maplist[1],minus_maplist[2]),axis=2)
    new_img = ori_img - minus_map
    
    return new_img

# draw normalized picture, so some pixels are negative --> need to add to all pixel positive
def drawface(avgimg, plt, savefn=''):
    avgimg -= np.min(avgimg) # min pixel over 3 channel, should it be changed to each channel in the future
    avgimg /= np.max(avgimg) # max pixel over 3 channel, should it be changed to each channel in the future
    saveimg = (avgimg*255).astype(np.uint8)
    # given filename
    if len(savefn)>0:
        plt.imsave(fname=savefn, arr=saveimg)
    else:
        plt.imshow(saveimg)
        plt.show()

    return

def main():
    parser = argparse.ArgumentParser(prog='ml2017fallhw6_PCA.py')
    parser.add_argument('--datadir',type=str,dest="datadir",default='data/Aberdeen/')
    parser.add_argument('--loadpicnp',type=str,default='')
    parser.add_argument('--train',type=bool,default=False)
    parser.add_argument('--newsize',type=int,default=60)
    parser.add_argument('--klargest',type=int,default=4)
    parser.add_argument('--saveU',type=str,default='pcapicU.npy')
    args = parser.parse_args()
    
    picflist = glob.glob(args.datadir+"/*.jpg")
    #print("picflist is {}".format(picflist))

    # no pre-rediced pictures np array loading
    if len(args.loadpicnp)==0:
        imgs = []
        for picf in picflist:
            img_origin = io.imread(picf)
            img_small = resizepic(img_origin, args)
            img_minus = minusavg(img_small)
            img_flat = img_minus.flatten()
            imgs.append(img_flat)

        #print("shape of small image is {}".format(img_small.shape))
        imgs = np.array(imgs, dtype=np.float32)
        print("shape of images array is {}".format(imgs.shape))
        np.save('aberdeen.npy', imgs)
        
    else:
        imgs = np.load(args.loadpicnp)

    # mean of each pixels (over 415 pictures)
    X_mean = np.transpose(np.mean(imgs, axis=0))
    X_mean = np.reshape(X_mean, (X_mean.shape[0],1))
    X = np.transpose(imgs)
    
    print("shape of X is {}".format(X.shape)) #checking shape after tranpose
    # SVD of X- Xmean
    X_ = np.subtract(X, X_mean)
    U, sigma, V = np.linalg.svd(X_, full_matrices=False)

    S = np.diag(sigma)

    #saving eigenvalues and eigenvectors into files
    np.save('PCAeigenvalue.npy',S)
    np.save(args.saveU, U)

    import  matplotlib.pyplot as plt
    # draw mean of picture
    print("Drawing and saving mean pixel of all pictures....")
    img_mean = np.reshape(X_mean, (args.newsize,args.newsize,3))
    drawface(img_mean,plt,'result/avgface.png')
    
    # draw k-largest eigenface
    k = args.klargest
    Uk = U[:,:k]
    for ien in range(k):
        Ui = np.squeeze(Uk[:,ien])
        eigenf = np.reshape(Ui,  (args.newsize,args.newsize,3))
        savefn = 'result/PCAeigenface%d.png' %(ien)
        print("Draw and save eigenface in file {}".format(savefn))
        drawface(eigenf,plt,savefn)
        
    return

if __name__ == '__main__':
    main()
