import numpy as np

def load_model():
    VOCAB_PATH = '../data/glove/vocab.txt'
    VEC_PATH = '../data/glove/vec.txt'
    vocab = {}
    for i,line in enumerate(open(VOCAB_PATH)):
        vocab[line.strip()] = i
    matrix = np.loadtxt(VEC_PATH)
    return vocab, matrix