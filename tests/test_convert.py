import random
from unittest.mock import patch
from word_vectors.read import read
from utils import vocab, vectors, DATA, GLOVE, W2V, DENSE, MAX_VOCAB


def test_read_convert():
    with patch("word_vectors.read_module.write_dense") as write_patch:
        w, wv = read(str(DATA / DENSE), convert=True, replace=False)
        write_patch.assert_not_called()


def test_read_convert_replace():
    with patch("word_vectors.read_module.write_dense") as write_patch:
        w, wv = read(str(DATA / DENSE), convert=True, replace=True)
        write_patch.assert_not_called()


def test_read_convert_opened():
    with patch("word_vectors.read_module.write_dense") as write_patch:
        w, wv = read(open(str(DATA / DENSE), "rb"), convert=True, replace=False)
        write_patch.assert_not_called()


def test_read_convert_replace_opened():
    with patch("word_vectors.read_module.write_dense") as write_patch:
        w, wv = read(open(str(DATA / DENSE), "rb"), convert=True, replace=True)
        write_patch.assert_not_called()


def test_read_convert_pathlib():
    with patch("word_vectors.read_module.write_dense") as write_patch:
        w, wv = read(DATA / DENSE, convert=True, replace=False)
        write_patch.assert_not_called()


def test_read_convert_replace_pathlib():
    with patch("word_vectors.read_module.write_dense") as write_patch:
        w, wv = read(DATA / DENSE, convert=True, replace=True)
        write_patch.assert_not_called()


def test_read_convert():
    data = random.choice([GLOVE, W2V])
    with patch("word_vectors.read_module.write_dense") as write_patch:
        w, wv = read(str(DATA / data), convert=True, replace=False)
        gold = str(DATA / data) + ".dense"
        write_patch.assert_called_once_with(gold, w, wv, MAX_VOCAB)


def test_read_convert_pathlib():
    data = random.choice([GLOVE, W2V])
    with patch("word_vectors.read_module.write_dense") as write_patch:
        w, wv = read(DATA / data, convert=True, replace=False)
        gold = str(DATA / data) + ".dense"
        write_patch.assert_called_once_with(gold, w, wv, MAX_VOCAB)


def test_read_convert_replace():
    data = random.choice([GLOVE, W2V])
    mode = "r" if data == GLOVE else "rb"
    with patch("word_vectors.read_module.write_dense") as write_patch:
        w, wv = read(open(DATA / data, mode), convert=True, replace=False)
        gold = str(DATA / data) + ".dense"
        write_patch.assert_called_once_with(gold, w, wv, MAX_VOCAB)


def test_read_convert_replace():
    data = random.choice([GLOVE, W2V])
    with patch("word_vectors.read_module.write_dense") as write_patch:
        w, wv = read(str(DATA / data), convert=True, replace=True)
        gold = str(DATA / data)
        write_patch.assert_called_once_with(gold, w, wv, MAX_VOCAB)


def test_read_convert_replace_pathlib():
    data = random.choice([GLOVE, W2V])
    with patch("word_vectors.read_module.write_dense") as write_patch:
        w, wv = read(DATA / data, convert=True, replace=True)
        gold = str(DATA / data)
        write_patch.assert_called_once_with(gold, w, wv, MAX_VOCAB)


def test_read_convert_replace():
    data = random.choice([GLOVE, W2V])
    mode = "r" if data == GLOVE else "rb"
    with patch("word_vectors.read_module.write_dense") as write_patch:
        w, wv = read(open(DATA / data, mode), convert=True, replace=True)
        gold = str(DATA / data)
        write_patch.assert_called_once_with(gold, w, wv, MAX_VOCAB)
