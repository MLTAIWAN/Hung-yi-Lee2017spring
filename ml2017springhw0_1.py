#!/bin/env python
#*-*coding=utf-8*-*

import sys
import numpy as np
#import csv

#using numpy function genfromtxt to read csv file into numpy array
matrix_A = np.genfromtxt('matrixA.txt',delimiter=',',skip_header=0,dtype=np.int32)
matrix_B = np.genfromtxt('matrixB.txt',delimiter=',',skip_header=0,dtype=np.int32)

matrix_A = np.reshape(matrix_A, (1,matrix_A.shape[0]))
print("Shape of matrixA is {} and the dtype is {}".format(matrix_A.shape,matrix_A.dtype))
print("Shape of matrixB is {} and the dtype is {}".format(matrix_B.shape,matrix_B.dtype))

#check the common element number before dot operation
if (matrix_A.shape[1]==matrix_B.shape[0]):
    matrix_C = np.dot(matrix_A,matrix_B)
    print("Shape of matrixC is {} and the dtype is {}".format(matrix_C.shape,matrix_C.dtype))
    print("Before sorting {}".format(matrix_C))
    matrix_C = np.reshape(np.sort(matrix_C, axis=1),(matrix_C.shape[1],matrix_C.shape[0]))
    print("After sorting and reshaping {}".format(matrix_C))
    np.savetxt('ans_one.txt',matrix_C,fmt="%4d",delimiter='')
    sys.exit(0)
else:
    print("Two matrices do not have common element number, exit")
    sys.exit(1)

