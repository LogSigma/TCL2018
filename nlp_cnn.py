from konlpy.tag import Twitter
from textcnn import TextCNN
import tensorflow as tf
import numpy as np

pos_tagger = Twitter()

def read_data(filename):
  with open(filename, 'r') as f:
    data = [line.split('\t') for line in f.read().splitlines()]
    
    data = [(tokenize(row[1]), int(row[2])) for row in data[1:]]
  return data

def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

def build_vocab(tokens):
    vocab = dict()
    vocab['#UNKOWN'] = 0
    vocab['#PAD'] = 1
    for t in tokens:
        if t not in vocab:
            vocab[t] = len(vocab)
    return vocab
  
def get_token_id(token, vocab):
    if token in vocab:
        return vocab[token]
    else:
        0

def build_input(data, vocab):
    def get_onehot(index, size):
        onehot = [0] * size
        onehot[index] = 1
        return onehot
    result = []
    for d in data:
        sequence = [get_token_id(t, vocab) for t in d[0]]
        while len(sequence) > 0:
            seq_seg = sequence[:60]
            sequence = sequence[60:]
            padding = [1] *(60 - len(seq_seg))
            seq_seg = seq_seg + padding
            result.append((seq_seg, get_onehot(d[1], 2)))
    return result 
  
# __main__
data = read_raw_data('ratings_train.txt')
tokens = [t for d in data for t in d[0]]
vocab = build_vocab(tokens)
d = build_input(data, vocab)
