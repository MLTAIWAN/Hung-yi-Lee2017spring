#!/bin/env python3
#*-* coding=utf-8 *-*            

import os,sys
import argparse
import glob
import numpy as np
from skimage import io, transform
from utils.util import resizepic, minusavg

def draw2face(img_origin, img_recon, plt, savefn=''):
    img_recon -= np.min(img_recon) 
    img_recon /= np.max(img_recon) 
    img_recon = (img_recon*255).astype(np.uint8)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))

    ca1 = ax1.imshow(img_origin, interpolation='nearest')
    frame = plt.gcf()
    title = 'Original pic:'
    ax1.set_title(title)

    ca2 = ax2.imshow(img_recon, interpolation='nearest')
    #cbar2 = fig.colorbar(ca2,ax=ax2)
    #plt.tight_layout()
    frame = plt.gcf()
    ax2.set_title('PCA reconstructed pic:')
    
    # given filename                  
    if len(savefn)>0:
        plt.savefig(savefn)
    else:
        #plt.imshow(saveimg)
        plt.show()

    return


def main():
    parser = argparse.ArgumentParser(prog='ml2017fallhw6_PCArecon.py')
    parser.add_argument('--datadir',type=str,dest="datadir",default='data/Aberdeen/')
    parser.add_argument('--picnumber',type=int,default=0)
    parser.add_argument('--newsize',type=int,default=60)
    parser.add_argument('--klargest',type=int,default=4)
    parser.add_argument('--loadU',type=str,default='pcapicU.npy')
    args = parser.parse_args()

    picfname = "%s/%d.jpg" %(args.datadir, args.picnumber)

    #checking the file existing or not
    if (os.path.exists(picfname)==False):
        print("The file {} is not found....".format(picfname))
        os.exit(0)
        return

    
    img_origin = io.imread(picfname)
    img_small = resizepic(img_origin, args)
    img_minus, img_minusmap = minusavg(img_small)
    img_flat = img_minus.flatten()
    
    print("Shape of flattened picture is {}".format(img_flat.shape))
    
    # load U matrix
    U = np.load(args.loadU)
    # checking dimension length of U with image dimension length
    if (U.shape[0]!=img_flat.shape[0]):
        print("Dimension length of U {} is not the same with the picture {}!".format(U.shape[0],img_flat.shape[0]))
        return
    # using k largest eigenface
    k = args.klargest
    Uk = U[:,:k]

    dot_prods = np.dot(Uk.T, img_flat)
    #print("shape of dot_prods is {}".format(dot_prods.shape))

    img_flat_recon = np.dot(dot_prods.T, Uk.T)    
    #print("shape of reconstructed flat is {}".format(img_flat_recon.shape))
    img_recon = np.reshape(img_flat_recon,  (args.newsize,args.newsize,3)) + img_minusmap

    resultfn = "result/PCApic_%d_k%d.png" %(args.picnumber,k)
    import matplotlib.pyplot as plt
    draw2face(img_small, img_recon, plt, resultfn)
    
    
    return

if __name__=='__main__':
    main()
