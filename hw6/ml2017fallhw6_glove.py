#!/usr/bin/env python3
#*-* coding=utf-8 *-* 

import argparse
import jieba
import pprint
import numpy as np
from adjustText import adjust_text
from collections import Counter
from nltk.corpus import stopwords
from glove import Corpus, Glove

import matplotlib as mpl
from utils.util import build_dict

def read_corpus(filename, stop, args):
    with open(filename, encoding='utf-8') as dataf:
        for line in dataf:
            ws_insen = jieba.cut(line, cut_all=args.jiebacut)
            
            yield [ws for ws in ws_insen if ws not in stop] 

def plot(Xs, Ys, Texts):
    import matplotlib.pyplot as plt
    font_name = "Source Han Sans TW"
    mpl.rcParams['font.family'] = font_name
    plt.plot(Xs, Ys, 'o')
    texts = [plt.text(X, Y, Text) for X, Y, Text in zip(Xs, Ys, Texts)]
    plt.title(str(adjust_text(texts, Xs, Ys, arrowprops=dict(arrowstyle='->', color='red'))))
    plt.show()
    

def wordvec(glove, word):
    try:
        word_idx = glove.dictionary[word]
    except KeyError:
        print("{} is not in corpus".format(word))
        return None
    
    return glove.word_vectors[word_idx]


def main():
    parser = argparse.ArgumentParser(prog='ml2017fallhw6_glove.py')
    parser.add_argument('--createcorpus',type=bool,default=False)
    parser.add_argument('--datafile',type=str,dest="datafile",default='data/all_sents.txt')
    parser.add_argument('--corpusfn',type=str,dest="corpusfn",default='glove_corpus.model')    
    parser.add_argument('--train',type=bool,default=False)
    parser.add_argument('-epochs', '--epochs', default=100, type=int)
    parser.add_argument('-nothread', '--nothread', default=4, type=int)
    parser.add_argument('-glovefn','--glovefn',type=str,dest="glovefn",default='glove_emb.model')
    parser.add_argument('-jiebacut','--jiebacut',type=bool,default=True)
    parser.add_argument('-mostk', '--mostk', default=200, type=int)
    args = parser.parse_args()
    
    
    jieba.initialize()
    jieba.set_dictionary('big5dict/dict.txt.big')

    stop = stopwords.words('chinese')
    stop.append("\n") #add wrap symbol
    
    if args.createcorpus:
        corpus_model = Corpus()
        corpus_model.fit(read_corpus(args.datafile, stop, args), window=10)
        corpus_model.save('glove_corpus.model')
        
        print('Dict size: %s' % len(corpus_model.dictionary))
        print('Collocations: %s' % corpus_model.matrix.nnz)
    
    if args.train:
        if not args.createcorpus:
            #load corpus from model file
            corpus_model = Corpus.load(args.corpusfn)
            print('Dict size: %s' % len(corpus_model.dictionary))
            print('Collocations: %s' % corpus_model.matrix.nnz)
 
        print("Word embedding training by glove-python")
        glove = Glove(no_components=200, learning_rate=0.017)
        glove.fit(corpus_model.matrix, epochs=int(args.epochs),
                  no_threads=args.nothread, verbose=args.jiebacut)
        glove.add_dictionary(corpus_model.dictionary)
        glove.save(args.glovefn)

    query_words = [u"問題",u"笑",u"放心",u"機構",u"告訴",u"學會"]
    if not args.train:
        print('Loading pre-trained Glove model')
        glove = Glove.load(args.glovefn)

    for qword in query_words:
        print("Closest words to {} are".format(qword))
        pprint.pprint(glove.most_similar(qword, number=10))

        
    # search most k frequent words (k=200 default)
    words = list()
    k = args.mostk
    with open(args.datafile, encoding='utf-8') as dataf:
        for line in dataf:
            ws_insen = jieba.cut(line, cut_all=args.jiebacut)
            ws_remove = [ws for ws in ws_insen
                         if ws not in stop]
            words.extend(ws_remove)

    countw, _, _ = build_dict(words, vocab_size=5000)
    kcounts = countw[1:k+1]
    kwords = list()
    kwords_vec = list()
    for word_, count_ in kcounts:
        kwords.append(word_)
        kwords_vec.append(wordvec(glove, word_))

    kwords_vecs = np.array(kwords_vec, dtype=np.float64)
    from sklearn.manifold import TSNE
    #preform t-SNE embedding
    vis_data = TSNE(n_components=2).fit_transform(kwords_vecs)
    Xs = vis_data[:,0]
    Ys = vis_data[:,1]

    plot(Xs, Ys, kwords)
    
    return

if __name__ == '__main__':
    main()
