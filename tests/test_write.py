import os
import random
import string
import pytest
from utils import vocab, vectors, DATA, GLOVE, W2V, DENSE
from word_vectors.write import write_glove, write_w2v, write_dense, _pad, to_vocab


@pytest.fixture
def file_name():
    _file_name = "temp"
    yield _file_name
    os.remove(_file_name)


@pytest.fixture
def w():
    return vocab


@pytest.fixture
def wv():
    return vectors


def test_to_vocab():
    vocab = list("ABCDEFGHIJKLMNOP")
    random.shuffle(vocab)
    d = {k: i for i, k in enumerate(vocab)}
    assert to_vocab(vocab) == d


def test_save_glove(w, wv, file_name):
    write_glove(file_name, w, wv)
    gold = open(DATA / GLOVE, "rb").read()
    mine = open(file_name, "rb").read()
    assert mine == gold


def test_save_w2v(w, wv, file_name):
    write_w2v(file_name, w, wv)
    gold = open(DATA / W2V, "rb").read()
    mine = open(file_name, "rb").read()
    assert mine == gold


def test_save_dense(w, wv, file_name):
    max_len = max(len(x) for x in w)
    write_dense(file_name, w, wv, max_len)
    gold = open(DATA / DENSE, "rb").read()
    mine = open(file_name, "rb").read()
    assert mine == gold


def test_pad():
    len_ = random.randint(5, 11)
    text = "".join(random.choice(string.ascii_lowercase) for _ in range(len_))
    len_ = random.randint(len(text), len(text) + 10)
    res = _pad(text, len_)
    assert len(res) == len_
    assert res[: len(text)] == text.encode("utf-8")


def test_pad_longer():
    len_ = random.randint(5, 11)
    text = "".join(random.choice(string.ascii_lowercase) for _ in range(len_))
    len_ -= 1
    res = _pad(text, len_)
    assert len(res) == len(text)
    assert len(res) > len_
    assert res == text.encode("utf-8")
