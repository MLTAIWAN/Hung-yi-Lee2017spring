#!/bin/env python3
#-*- coding=utf-8 -*-

import numpy as np
import pandas as pd

def batch_index(batch_size, data_size):
    # Create batches for each epoch                                                           
    num_batches = int(data_size/batch_size) + 1
    # Split up text indices into subarrays, of equal size                                     
    batches = np.array_split(train_ix, num_batches)
    # Reshape each split into [batch_size, training_seq_len]                                  
    batches = [np.resize(x, [batch_size]) for x in batches]

    return batches

def get_batches(x, y, batch_size=100):
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]

def get_batches_nolabel(x, batch_size=100):
    n_batches = len(x)//batch_size+1
    x = x[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        if (ii+batch_size<len(x)):
            yield x[ii:ii+batch_size]
        else:
            repeat_idx = ii+batch_size-len(x)
            yield np.vstack((x[ii:ii+batch_size],x[:repeat_idx]))
    
