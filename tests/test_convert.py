import os
import random
import pathlib
from unittest.mock import patch
import numpy as np
from word_vectors import FileType
from word_vectors.read import read
from word_vectors.convert import convert
from utils import vocab, vectors, DATA, GLOVE, W2V, W2V_TEXT, DENSE, MAX_VOCAB, rand_str


INPUT_MAPPING = {
    GLOVE: FileType.GLOVE,
    W2V: FileType.W2V,
    W2V_TEXT: FileType.W2V_TEXT,
    DENSE: FileType.DENSE,
}


def test_convert():
    data = random.choice([GLOVE, W2V, W2V_TEXT, DENSE])
    output_type = random.choice(list(FileType))
    input_path = str(DATA / data)
    gold_output_path = os.path.splitext(input_path)[0] + "." + str(output_type)
    with patch("word_vectors.convert_module.write") as write_patch:
        w, wv = read(input_path)
        convert(input_path, output_file_type=output_type)
        call_file, call_w, call_wv, call_type, call_max = write_patch.call_args_list[0][0]
        assert call_file == gold_output_path
        assert call_w == w
        np.testing.assert_allclose(call_wv, wv)
        assert call_type is output_type
        assert call_max == MAX_VOCAB


def test_convert_with_output():
    data = random.choice([GLOVE, W2V, W2V_TEXT, DENSE])
    output_type = random.choice(list(FileType))
    output = rand_str()
    input_path = str(DATA / data)
    with patch("word_vectors.convert_module.write") as write_patch:
        w, wv = read(input_path)
        convert(input_path, output, output_file_type=output_type)
        call_file, call_w, call_wv, call_type, call_max = write_patch.call_args_list[0][0]
        assert call_file == output
        assert call_w == w
        assert call_type == output_type
        np.testing.assert_allclose(call_wv, wv)
        assert call_max == MAX_VOCAB


def test_convert_with_input():
    data = random.choice([GLOVE, W2V, W2V_TEXT, DENSE])
    input_type = INPUT_MAPPING[data]
    output_type = random.choice(list(FileType))
    input_path = str(DATA / data)
    output = rand_str()
    with patch("word_vectors.convert_module.read") as read_patch:
        with patch("word_vectors.convert_module.write") as write_patch:
            w, wv = read(input_path)
            read_patch.return_value = (w, wv)
            convert(input_path, output, output_file_type=output_type, input_file_type=input_type)
            read_patch.assert_called_once_with(input_path, input_type)
            call_file, call_w, call_wv, call_type, call_max = write_patch.call_args_list[0][0]
            assert call_file == output
            assert call_w == w
            assert call_type == output_type
            np.testing.assert_allclose(call_wv, wv)
            assert call_max == MAX_VOCAB


def test_convert_pathlib():
    data = random.choice([GLOVE, W2V, W2V_TEXT, DENSE])
    output_type = random.choice(list(FileType))
    input_path = DATA / data
    gold_output_path = os.path.splitext(str(input_path))[0] + "." + str(output_type)
    with patch("word_vectors.convert_module.write") as write_patch:
        w, wv = read(input_path)
        convert(input_path, output_file_type=output_type)
        call_file, call_w, call_wv, call_type, call_max = write_patch.call_args_list[0][0]
        assert str(call_file) == gold_output_path
        assert call_w == w
        np.testing.assert_allclose(call_wv, wv)
        assert call_type is output_type
        assert call_max == MAX_VOCAB


def test_convert_with_output_pathlib():
    data = random.choice([GLOVE, W2V, W2V_TEXT, DENSE])
    output_type = random.choice(list(FileType))
    output = pathlib.Path(rand_str())
    input_path = DATA / data
    with patch("word_vectors.convert_module.write") as write_patch:
        w, wv = read(input_path)
        convert(input_path, output, output_file_type=output_type)
        call_file, call_w, call_wv, call_type, call_max = write_patch.call_args_list[0][0]
        assert call_file == output
        assert call_w == w
        assert call_type == output_type
        np.testing.assert_allclose(call_wv, wv)
        assert call_max == MAX_VOCAB


def test_convert_with_input_pathlib():
    data = random.choice([GLOVE, W2V, W2V_TEXT, DENSE])
    input_type = INPUT_MAPPING[data]
    output_type = random.choice(list(FileType))
    input_path = DATA / data
    output = pathlib.Path(rand_str())
    with patch("word_vectors.convert_module.read") as read_patch:
        with patch("word_vectors.convert_module.write") as write_patch:
            w, wv = read(input_path)
            read_patch.return_value = (w, wv)
            convert(input_path, output, output_file_type=output_type, input_file_type=input_type)
            read_patch.assert_called_once_with(input_path, input_type)
            call_file, call_w, call_wv, call_type, call_max = write_patch.call_args_list[0][0]
            assert call_file == output
            assert call_w == w
            assert call_type == output_type
            np.testing.assert_allclose(call_wv, wv)
            assert call_max == MAX_VOCAB


def test_convert_open():
    data = random.choice([GLOVE, W2V, W2V_TEXT, DENSE])
    output_type = random.choice(list(FileType))
    input_path = DATA / data
    gold_output_path = os.path.splitext(str(input_path))[0] + "." + str(output_type)
    with open(input_path, "r" if data in (GLOVE, W2V_TEXT) else "rb") as input_path:
        with patch("word_vectors.convert_module.write") as write_patch:
            w, wv = read(input_path)
            convert(input_path, output_file_type=output_type)
            call_file, call_w, call_wv, call_type, call_max = write_patch.call_args_list[0][0]
            assert str(call_file) == gold_output_path
            assert call_w == w
            np.testing.assert_allclose(call_wv, wv)
            assert call_type is output_type
            assert call_max == MAX_VOCAB


def test_convert_with_output_open():
    data = random.choice([GLOVE, W2V, W2V_TEXT, DENSE])
    output_type = random.choice(list(FileType))
    output = rand_str()
    input_path = DATA / data
    with open(input_path, "r" if data in (GLOVE, W2V_TEXT) else "rb") as input_path:
        with open(output, "w" if output_type in (FileType.GLOVE, FileType.W2V_TEXT) else "wb") as output:
            with patch("word_vectors.convert_module.write") as write_patch:
                w, wv = read(input_path)
                convert(input_path, output, output_file_type=output_type)
                call_file, call_w, call_wv, call_type, call_max = write_patch.call_args_list[0][0]
                assert call_file == output
                assert call_w == w
                assert call_type == output_type
                np.testing.assert_allclose(call_wv, wv)
                assert call_max == MAX_VOCAB


def test_convert_with_input_open():
    data = random.choice([GLOVE, W2V, W2V_TEXT, DENSE])
    input_type = INPUT_MAPPING[data]
    output_type = random.choice(list(FileType))
    input_path = DATA / data
    output = rand_str()
    with open(input_path, "r" if data in (GLOVE, W2V_TEXT) else "rb") as input_path:
        with open(output, "w" if output_type in (FileType.GLOVE, FileType.W2V_TEXT) else "wb") as output:
            with patch("word_vectors.convert_module.read") as read_patch:
                with patch("word_vectors.convert_module.write") as write_patch:
                    w, wv = read(input_path)
                    read_patch.return_value = (w, wv)
                    convert(input_path, output, output_file_type=output_type, input_file_type=input_type)
                    read_patch.assert_called_once_with(input_path, input_type)
                    call_file, call_w, call_wv, call_type, call_max = write_patch.call_args_list[0][0]
                    assert call_file == output
                    assert call_w == w
                    assert call_type == output_type
                    np.testing.assert_allclose(call_wv, wv)
                    assert call_max == MAX_VOCAB
