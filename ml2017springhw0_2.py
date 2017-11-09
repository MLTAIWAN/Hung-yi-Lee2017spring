#!/usr/bin/env python
#*-*coding=utf-8*-*

#ml2017springhw0_2.py
#write by Kunxian Huang
#9th, Nov., 2017

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

fName_lena = "lena.jpeg"
fName_lena_modified = "lena_modified.jpeg"

im_lena = Image.open(fName_lena).convert('RGB')
im_lena_mod = Image.open(fName_lena_modified).convert('RGB')


arr_lena = np.array(im_lena)
print(arr_lena.shape)
arr_lena_mod = np.array(im_lena_mod)
print(arr_lena_mod.shape)
print("dtype of the array is", arr_lena_mod.dtype)

#def fncom(a,b):
#    if a==b:
#        return 0
#    else:
#        return a

#using map to make new np.array
#but it does not work for this, why?
#arr_willy = np.fromiter(map(fncom, arr_lena_mod,arr_lena), dtype=arr_lena_mod.dtype)

#So, I have to put item value by loop
arr_willy = np.zeros(arr_lena_mod.shape, dtype=arr_lena_mod.dtype)
print(arr_willy.shape)
for i in range(arr_willy.shape[0]):
    for j in range(arr_willy.shape[1]):
        for k in range(arr_willy.shape[2]):
            if (arr_lena_mod[i][j][k]==arr_lena[i][j][k]):
                arr_willy.itemset((i,j,k), 0)
            else:
                #print("Different elements %i %i " %(arr_lena_mod[i][j][k],arr_lena[i][j][k]))
                arr_willy.itemset((i,j,k), arr_lena_mod[i][j][k])
            
#PIL show
#im_willy = Image.fromarray(arr_willy, 'RGB')
#im_willy.show()

#matplotlib show
plt.imshow(arr_willy, interpolation='nearest')
plt.show()
#plt.imsave("willy.png")
