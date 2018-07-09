#!/bin/env python3
#-*- coding=utf-8 -*-
#make word embedding by using all data (train, semi, test)

import os, sys
import collections
from itertools import compress
import argparse
import random
import _pickle as pk
import tensorflow as tf
import numpy as np
from utils.util import DataManager

def generate_batch_cbow(data, data_index, seq_index, batch_size, num_skips=2, skip_window=1):
    """
    generate cbow batch for training (Continuous Bag of Words).
    batch shape: (batch_size, num_skips)
    label shape: (batch_size, 1)
    Parameters
    ----------
    data: list of index of words, then sentence
    batch_size: number of words in each mini-batch
    num_skips: number of surrounding words on both direction (2: one word ahead and one word following)
    skip_window: number of words at both ends of a sentence to skip (1: skip the first and last word of a sentence)
    """
    #assert batch_size % num_skips == 0
    #assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size, num_skips), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span) # used for collecting data[data_index] in the sliding window
    # check the length of sequence every before first collecting
    while (len(data[data_index][seq_index:])<span):
        seq_index = 0
        data_index = (data_index+1) %len(data)
    
    # collect the first window of words
    for ispan in range(span):
        try:
            #append one word token each time
            if (isinstance(data[data_index][seq_index], int)==False):
                print("{} is not a single word, why??\n".format(data[data_index][seq_index]))
                print("data index {}, seq_index {}".format(data_index, seq_index))
                
            buffer.append(data[data_index][seq_index])
            seq_index += 1
        except IndexError:
            print("Length of sentence is {}, and we are at index of {}".format(len(data[data_index]), seq_index))

    # move the sliding window  
    for i in range(batch_size):
        mask = [1 for _,i in enumerate(range(span))]
        mask[skip_window] = 0 
        batch[i, :] = list(compress(buffer, mask)) # all surrounding words
        labels[i, 0] = buffer[skip_window] # the word at the center
        if (len(data[data_index])>seq_index):
            if (isinstance(data[data_index][seq_index], int)==False):
                print("{} is not a single word, why??".format(data[data_index][seq_index]))
                print("data index {}, seq_index {}".format(data_index, seq_index))
            buffer.append(data[data_index][seq_index])
            seq_index += 1
        else:
            #jump to next sentence
            seq_index = 0
            # check the sequnce length is larger than span
            seq_len = 0
            while (seq_len<span):
                data_index = (data_index + 1) % len(data)
                seq_len = len(data[data_index])
                
            for _ in range(span):
                if (isinstance(data[data_index][seq_index], int)==False):
                    print("{} is not a single word, why??".format(data[data_index][seq_index]))
                    print("data index {}, seq_index {}".format(data_index, seq_index))
                    
                buffer.append(data[data_index][seq_index])
                seq_index += 1
                
    return batch, labels, data_index, seq_index


def main():
    parser = argparse.ArgumentParser(description='CBOW word embedding')
    #training argument
    parser.add_argument('--vocab_size', default=20000, type=int)
    parser.add_argument('-emb_dim', '--embedding_dim', default=128, type=int)
    parser.add_argument('--gpu_fraction', default=0.4, type=float)
    parser.add_argument('--skip_window', default=1, type=int)
    parser.add_argument('--num_skips', default=2, type=int)
    parser.add_argument('--batch_size', default=384, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--log_dir', default='log_embdir/')
    parser.add_argument('--nsteps', default=700000, type=int)
    
    # put model in the same directory                                                        
    parser.add_argument('--load_model', default = None)
    parser.add_argument('--load_token', default = None, type=bool)
    parser.add_argument('--save_embed', default = 'cbowemb.npz')
    
    args = parser.parse_args()
    
    train_path = 'data/training_label.txt'
    test_path = 'data/testing_data.txt'
    semi_path = 'data/training_nolabel.txt'
    save_path = 'token/'

    #load token path                                                    
    if args.load_token is not None:
        load_path = os.path.join(save_path)

       # limit gpu memory usage                                                                  
    def get_session(gpu_fraction):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    sess = get_session(args.gpu_fraction)

    #read all data for tokenizer (train, semi, test)
    dm = DataManager()
    print('Loading training data...')
    dm.add_data('train_data', train_path, True)
    dm.add_data('semi_data', semi_path, False)
    dm.add_data('test_data', test_path, False)

    # prepare tokenizer                                                                       
    print ('get Tokenizer...')
    if args.load_token is not None:
        # read exist tokenizer
        dm.load_tokenizer(os.path.join(load_path,'token.pk'))
    else:
        # create tokenizer on new data
        dm.tokenize(args.vocab_size)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path,'token.pk')):
        dm.save_tokenizer(os.path.join(save_path,'token.pk'))


    # prepare sequence to text dict
    reverse_word_dict = dict(map(reversed, dm.tokenizer.word_index.items()))
    
    # CBOW embedding [skip_window target skip_window]
    context_size = args.skip_window*2 

    # convert to sequences without pre-padding (list, not np.array)
    #dm.to_sequence(args.max_length)
    dm.to_sequence_nopad()

    # fill all sequence data into a list
    seq_data = []
    seq_data.extend(dm.get_data('train_data')[0])
    seq_data.extend(dm.get_data('semi_data')[0])
    seq_data.extend(dm.get_data('test_data')[0])
    
    # Create the graph object
    tf.reset_default_graph()

    # pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent. These 3 variables are used only for
    # displaying model accuracy, they don't affect calculation.
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    #valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    valid_examples = np.array(random.sample(range(valid_window), valid_size))

    with tf.name_scope('inputs'):
        #create placeholder for training (testing) data                         
        X_ = tf.placeholder(tf.int32, [args.batch_size, args.num_skips], name='X_')
        y_ = tf.placeholder(tf.int32, [args.batch_size, 1], name='y_')
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        
    #embedding here
    with tf.name_scope("embeddings"):
        embedding_mat = tf.get_variable('embedding_mat', [args.vocab_size, args.embedding_dim],
                                     tf.float32, tf.random_normal_initializer())
        #embedding num_skips words
        embedding = tf.zeros([args.batch_size, args.embedding_dim])
        for j in range(args.num_skips):
            embedding += tf.nn.embedding_lookup(embedding_mat, X_[:,j])

    with tf.name_scope("softmax"):
        soft_weights = tf.get_variable('soft_weights', [args.vocab_size, args.embedding_dim],
                                       tf.float32, tf.random_normal_initializer())
        soft_biases = tf.get_variable('soft_biases', [args.vocab_size], tf.float32, tf.constant_initializer(0.0))

    num_sampled = 64
    # Compute the loss
    with tf.name_scope('loss'):
        # tf.nn.nce_loss
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=soft_weights, biases=soft_biases,
                                             labels=y_, inputs=embedding,
                                             num_sampled=num_sampled,
                                             num_classes=args.vocab_size))
        # Add the loss value as a scalar to summary.
        tf.summary.scalar('loss', loss)
    with tf.name_scope('optimizer'):
        optimizer=tf.train.AdagradOptimizer(args.learning_rate).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding_mat),1, keep_dims=True))
    #normalized embedding matrix by its summation of squre element value, then take squre root
    normalized_embeddings = embedding_mat/norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

    # Merge all summaries.
    merged = tf.summary.merge_all()

    # variable initializer
    init = tf.initialize_all_variables()
    
    #tensorflow model saver
    saver = tf.train.Saver(tf.global_variables())

    writer = tf.summary.FileWriter(args.log_dir, sess.graph)

    average_loss = 0.0
    data_index=0
    seq_index=0
    with tf.Session() as sess:
        # start to training
        sess.run(init)
        for step in range(args.nsteps):
            batch_X, batch_y, data_index, seq_index = generate_batch_cbow(seq_data,data_index,seq_index,
                                                                          args.batch_size, args.num_skips, args.skip_window)
            feed_dict = {X_: batch_X, y_: batch_y}
            op, lo = sess.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += lo
            if (step % 2000 == 0):
                if (step > 0):
                    average_loss = average_loss / 2000
		    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step %d: %f' % (step, average_loss))
                    average_loss = 0

            # note that this is expensive (~20% slowdown if computed every 500 steps)
            if (step % 10000 == 0):
                sim = similarity.eval()
                for i in range(valid_size):
                    try:
                        valid_word = reverse_word_dict[valid_examples[i]]
                    except KeyError:
                        print("Skip word...")
                        
                    top_k = 8 # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        try:
                            close_word = reverse_word_dict[nearest[k]]
                            log = '%s %s,' % (log, close_word)
                        except KeyError:
                            print("Skip nearest {}-th word".format(k))
                    #print once for each word    
                    print(log)
    
        
        # final_embeddings = self.normalized_embeddings.eval()
        #final_embeddings = normalized_embeddings.eval()
        final_embeddings = embedding_mat.eval()
        # Save the model for checkpoints.
        saver.save(sess, os.path.join(args.log_dir, 'embmodel.ckpt'))
    
        writer.close()

    #save the embedding mapping matrix
    save_fn = save_path+args.save_embed
    np.savez(save_fn, embed_m=final_embeddings)
    
    
    return

if  __name__ == "__main__":
    main()
    
