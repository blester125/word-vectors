from pathlib import Path
import pytest
import numpy as np
from word_vectors.read import read, read_glove, read_w2v, read_dense, sniff, Vectors, find_max
from utils import vocab, vectors, DATA, GLOVE, W2V, DENSE

@pytest.fixture
def gold_vocab():
    return vocab

@pytest.fixture
def gold_vectors():
    return vectors

def test_read_glove(gold_vocab, gold_vectors):
    w, wv = read_glove(DATA / GLOVE)
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)

def test_read_w2v(gold_vocab, gold_vectors):
    w, wv = read_w2v(DATA / W2V)
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)

def test_read_dense(gold_vocab, gold_vectors):
    w, wv = read_dense(DATA / DENSE)
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)

def test_sniff_glove():
    x = sniff(DATA / GLOVE)
    assert x is Vectors.GLOVE

def test_sniff_w2v():
    x = sniff(DATA / W2V)
    assert x is Vectors.W2V

def test_sniff_dense():
    x = sniff(DATA / DENSE)
    assert x is Vectors.DENSE

def test_max():
    words = ["A" * i for i in range(10)]
    gold_len = 9
    max_len = find_max(words)
    assert max_len == gold_len
