import random
from unittest.mock import patch
import numpy as np
from word_vectors.read import read
from word_vectors.convert import convert
from utils import vocab, vectors, DATA, GLOVE, W2V, W2V_TEXT, DENSE, MAX_VOCAB, rand_str


def test_convert():
    data = random.choice([GLOVE, W2V, W2V_TEXT])
    with patch("word_vectors.convert_module.write_dense") as write_patch:
        w, wv = read(str(DATA / data))
        convert(str(DATA / data))
        gold = str(DATA / data) + ".dense"
        call_file, call_w, call_wv, call_max = write_patch.call_args_list[0][0]
        assert call_file == gold
        assert call_w == w
        np.testing.assert_allclose(call_wv, wv)
        assert call_max == MAX_VOCAB

        
def test_convert_with_output():
    data = random.choice([GLOVE, W2V, W2V_TEXT])
    output = rand_str()
    with patch("word_vectors.convert_module.write_dense") as write_patch:
        w, wv = read(str(DATA / data))
        convert(str(DATA / data), output)
        call_file, call_w, call_wv, call_max = write_patch.call_args_list[0][0]
        assert call_file == output
        assert call_w == w
        np.testing.assert_allclose(call_wv, wv)
        assert call_max == MAX_VOCAB


def test_convert_pathlib():
    data = random.choice([GLOVE, W2V, W2V_TEXT])
    with patch("word_vectors.convert_module.write_dense") as write_patch:
        w, wv = read(str(DATA / data))
        convert(str(DATA / data))
        gold = str(DATA / data) + ".dense"
        call_file, call_w, call_wv, call_max = write_patch.call_args_list[0][0]
        assert call_file == gold
        assert call_w == w
        np.testing.assert_allclose(call_wv, wv)
        assert call_max == MAX_VOCAB

        
def test_convert_pathlib_with_output():
    data = random.choice([GLOVE, W2V, W2V_TEXT])
    output = rand_str()
    with patch("word_vectors.convert_module.write_dense") as write_patch:
        w, wv = read(str(DATA / data))
        convert(str(DATA / data), output)
        call_file, call_w, call_wv, call_max = write_patch.call_args_list[0][0]
        assert call_file == output
        assert call_w == w
        np.testing.assert_allclose(call_wv, wv)
        assert call_max == MAX_VOCAB




def test_convert_open():
    data = random.choice([GLOVE, W2V, W2V_TEXT])
    with patch("word_vectors.convert_module.write_dense") as write_patch:
        w, wv = read(str(DATA / data))
        convert(str(DATA / data))
        gold = str(DATA / data) + ".dense"
        call_file, call_w, call_wv, call_max = write_patch.call_args_list[0][0]
        assert call_file == gold
        assert call_w == w
        np.testing.assert_allclose(call_wv, wv)
        assert call_max == MAX_VOCAB

        
def test_convert_open_with_output():
    data = random.choice([GLOVE, W2V, W2V_TEXT])
    output = rand_str()
    with patch("word_vectors.convert_module.write_dense") as write_patch:
        w, wv = read(str(DATA / data))
        convert(str(DATA / data), output)
        call_file, call_w, call_wv, call_max = write_patch.call_args_list[0][0]
        assert call_file == output
        assert call_w == w
        np.testing.assert_allclose(call_wv, wv)
        assert call_max == MAX_VOCAB

