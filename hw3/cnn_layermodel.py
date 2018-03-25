#!/usr/bin/env python3
#-*-coding=utf-8-*-

import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm

A = 1.7159 #Amplitude of activation function 
pwidth = 48
pheight = 48


#CNN model for emotion classification
class cnn_model(object):
    #initialize
    def __init__(self, conv_w1=5, conv_h1=5, conv_w2=3, conv_h2=3, pool_w=3, pool_h=3, pool1="max", pool3="avg", pool5="avg", n_filter=[64,64,64,128,128],
                 n_hidlayer=4, n_hidden=[1024, 1024, 128], target_size=7, learning_rate=0.037, l2=0.001, stride_w=2, stride_h=2, keep_drop=0.6):
        self.conv_w1 = conv_w1
        self.conv_h1 = conv_h1
        self.conv_w2 = conv_w2
        self.conv_h2 = conv_h2
        self.pool_w = pool_w
        self.pool_h = pool_h
        self.stride_w = stride_w
        self.stride_h = stride_h
        self.pool1   = pool1
        self.pool3   = pool3
        self.pool5   = pool5
        self.n_filter = n_filter
        #self.n_conv = n_conv
        self.n_hidlayer = n_hidlayer
        self.n_hidden = n_hidden
        self.num_channels = 1 #grey scale
        self.target_size = target_size #
        self.learning_rate = learning_rate
        self.l2 = l2
        self.keep_drop = keep_drop
                    
    def model(self, input_images, size, training):
        with tf.name_scope("cnn"):
            he_init = tf.contrib.layers.variance_scaling_initializer()
            
            #first convolution layer, stride=1X1, activation fn of relu
            bn_params = {        'is_training': training,
                                 'decay': 0.99,
                                 'updates_collections': None,
                                 'scale': True}
            conv1 = tf.contrib.layers.conv2d(input_images, self.n_filter[0], [self.conv_h1, self.conv_w1],
                                             stride=1, activation_fn=tf.nn.relu, weights_initializer=he_init,
                                             normalizer_fn=batch_norm, normalizer_params=bn_params, padding='SAME')
            conv1_drop = tf.contrib.layers.dropout(conv1, self.keep_drop, is_training=training)

            #first pooling layer
            #avg pool or max pool
            if (self.pool1=="avg"):
                pool1 = tf.contrib.layers.avg_pool2d(conv1_drop, kernel_size=[self.pool_h, self.pool_w],
                                       stride=[self.stride_h, self.stride_w], padding='SAME')
            elif(self.pool1=="max"):
                pool1 = tf.contrib.layers.max_pool2d(conv1_drop, kernel_size=[self.pool_h, self.pool_w],
                                       stride=[self.stride_h, self.stride_w], padding='SAME')

            #local response normalization
            #norm1 = tf.nn.lrn(pool1, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm1')
            
            #second convolution layer, stride=1X1, activation fn of elu           
            conv2 = tf.contrib.layers.conv2d(pool1, self.n_filter[1], [self.conv_h1, self.conv_w1],
                                             stride=1, activation_fn=tf.nn.relu, weights_initializer=he_init,
                                             normalizer_fn=batch_norm, normalizer_params=bn_params, padding='SAME')

            conv2_drop = tf.contrib.layers.dropout(conv2, self.keep_drop, is_training=training)

            #third convolution layer, stride=1X1, activation fn of elu
            conv3 = tf.contrib.layers.conv2d(conv2_drop, self.n_filter[2], [self.conv_h1, self.conv_w1],
                                             stride=1, activation_fn=tf.nn.relu, weights_initializer=he_init,
                                             normalizer_fn=batch_norm, normalizer_params=bn_params,
                                             padding='SAME')

            conv3_drop = tf.contrib.layers.dropout(conv3, self.keep_drop, is_training=training)
            
            #second pooling layer
            #avg pool or max pool
            if (self.pool3=="avg"):
                pool3 = tf.contrib.layers.avg_pool2d(conv3_drop, kernel_size=[self.pool_h, self.pool_w],
                                       stride=[self.stride_h, self.stride_w], padding='SAME')
            elif(self.pool3=="max"):
                pool3 = tf.contrib.layers.max_pool2d(conv3_drop, kernel_size=[self.pool_h, self.pool_w],
                                       stride=[self.stride_h, self.stride_w], padding='SAME')

            #forth convolution layer, stride=1X1, activation fn of elu
            conv4 = tf.contrib.layers.conv2d(pool3, self.n_filter[3], [self.conv_h2, self.conv_w2],
                                             stride=1, activation_fn=tf.nn.relu, weights_initializer=he_init,
                                             normalizer_fn=batch_norm, normalizer_params=bn_params,
                                             padding='SAME')

            conv4_drop = tf.contrib.layers.dropout(conv4, self.keep_drop, is_training=training)

            #fifth convolution layer, stride=1X1, activation fn of elu
            conv5 = tf.contrib.layers.conv2d(conv4_drop, self.n_filter[4], [self.conv_h2, self.conv_w2],
                                             stride=1, activation_fn=tf.nn.relu, weights_initializer=he_init,
                                             normalizer_fn=batch_norm, normalizer_params=bn_params,
                                             padding='SAME')

            conv5_drop = tf.contrib.layers.dropout(conv5, self.keep_drop, is_training=training)
            
            #avg pool or max pool
            if (self.pool5=="avg"):
                pool5 = tf.contrib.layers.avg_pool2d(conv5_drop, kernel_size=[self.pool_h, self.pool_w],
                                       stride=[self.stride_w, self.stride_h], padding='SAME')
            elif(self.pool5=="max"):
                pool5 = tf.contrib.layers.max_pool2d(conv5_drop, kernel_size=[self.pool_w, self.pool_h],
                                       stride=[self.stride_h, self.stride_w], padding='SAME')

            #flaten the layer into a 1D array for the following fully connect layers
            # flaten
            flat_out = tf.contrib.layers.flatten(pool5)
            
            #fully_connected layer 1: fully connect layer, activation function of elu
            fully_connected1 = tf.contrib.layers.fully_connected(flat_out, self.n_hidden[0],
                                                                 activation_fn=tf.nn.relu, weights_initializer=he_init,
                                                                 normalizer_fn=batch_norm, normalizer_params=bn_params)

            fully_connected1_drop = tf.contrib.layers.dropout(fully_connected1, self.keep_drop,is_training=training)
            
            #fully_connected layer 2: fully connect layer, activation function of elu
            fully_connected2 = tf.contrib.layers.fully_connected(fully_connected1_drop, self.n_hidden[1],
                                                                 activation_fn=tf.nn.relu, weights_initializer=he_init,
                                                                 normalizer_fn=batch_norm, normalizer_params=bn_params)
            
            fully_connected2_drop = tf.contrib.layers.dropout(fully_connected2, self.keep_drop,is_training=training)
            #fully_connected layer 3: fully connect layer, activation function of elu
            fully_connected3 = tf.contrib.layers.fully_connected(fully_connected2_drop, self.n_hidden[2],
                                                                 activation_fn=tf.nn.relu, weights_initializer=he_init,
                                                                 normalizer_fn=batch_norm, normalizer_params=bn_params)
            
            fully_connected3_drop = tf.contrib.layers.dropout(fully_connected3, self.keep_drop,is_training=training)

            #model output
            model_output = tf.contrib.layers.fully_connected(fully_connected3_drop, self.target_size,
                                                             activation_fn=None, weights_initializer=he_init,
                                                             normalizer_fn=None)
            
            #softmax output
            logit_output = tf.contrib.layers.softmax(model_output)
            predictions = logit_output
            #if validation or prediction
        #if (training_logi ==tf.estimator.ModeKeys.PREDICT):
        #    return tf.estimator.EstimatorSpec(mode=training_logi, predictions=predictions)
            
        return logit_output

    def loss_ce(self, logits, targets):
        #with tf.name_scope("loss"):
        #get rid of extra dimensions and cast targets into integers
        #targets = tf.squeeze(tf.cast(targets, tf.int32))
        targets = tf.squeeze(targets)
        logits  = tf.squeeze(logits)
        #calculate cross entropy, however sparse_softmax_cross_entropy_with_logits has problem about label dimemsion
        #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
        #calculate cross entropy, try tf.nn.softmax_cross_entropy_with_logits
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
        #reduce mean over batch_size
        mean_ce = tf.reduce_mean(cross_entropy)
        #L2 regularization loss
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #add two loss
        loss_ = tf.add_n([mean_ce] + reg_losses)
        #return loss_
        return mean_ce

    def optimizer(self, loss, gene_num):
        lr_decay = 0.1
        num_gens_to_wait = 20000
        
        model_learning_rate = tf.train.exponential_decay(self.learning_rate, gene_num,
                                                         num_gens_to_wait ,lr_decay, staircase=True)
        #optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        #optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,momentum=0.9, decay=0.9, epsilon=1e-10)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate, rho=0.95, epsilon=1e-08)
        #optimizer = tf.train.MomentumOptimizer(learning_rate=model_learning_rate,
        #                                       momentum=0.9, use_nesterov=True)
        #training_op = optimizer.minimize(loss, global_step=gene_num)
        training_op = optimizer.minimize(loss)
        return training_op
    
    def accuracy(self, logits, targets):
        #get rid of extra dimensions and cast targets into integers
        targets = tf.squeeze(tf.cast(targets, tf.int32))
        #prediction tf.argmax return the index
        predictions = tf.cast(tf.argmax(tf.squeeze(logits),1), tf.int32)
        target_index = tf.cast(tf.argmax(targets,1), tf.int32)
        #vector for prediction correctly or not
        predict_correct = tf.equal(predictions, target_index)
        #reduce to mean value or accuracy
        accuracy = tf.reduce_mean(tf.cast(predict_correct, tf.float32))
        return accuracy
