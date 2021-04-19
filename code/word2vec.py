import string;
import re;
import random
import math
import numpy as np
from gensim.models import Word2Vec
import pickle as pkl
from itertools import *
embed_d = 128
window_s = 5


def load_data(path="../dataset/paper_abstract.pkl", maxlen = None, n_words = 600000, sort_by_len = False):
    f = open(path, 'rb')
    content_set = pkl.load(f)
    f.close()

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    content_set_x, content_set_y = content_set

    content_set_x = remove_unk(content_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(content_set_x)
        content_set_x = [content_set_x[i] for i in sorted_index]

    return content_set_x


def word2vec():
    data = load_data()
    model = Word2Vec(sentences=data, window=5, vector_size=128, workers=3, sg=1, epochs=100)
    word_embed = np.zeros((32786, 128))
    for index, word in enumerate(model.wv.index_to_key):
        # Word embeddding vector stored at index=wordID of embedding array
        word_embed[int(word)] = model.wv[word]
    len = word_embed.shape[0]
    np.savetxt('../dataset/word_embeddings1.txt', word_embed[0:int(len/4)], delimiter=' ')
    np.savetxt('../dataset/word_embeddings2.txt', word_embed[int(len/4):int(len/2)], delimiter=' ')
    np.savetxt('../dataset/word_embeddings3.txt', word_embed[int(len/2):int(3*len/4)], delimiter=' ')
    np.savetxt('../dataset/word_embeddings4.txt', word_embed[int(3*len/4):], delimiter=' ')
    # generate word embedding files: word_embeddings1.txt and word_embeddings2.txt
    # Split embeddings into 4 files to comply with Github's file size recommendation of 50mb per file

#For testing
if __name__ == "__main__":

    model = word2vec()
