import os
import random
import string
from unittest.mock import patch
import pytest
from utils import vocab, vectors, DATA, GLOVE, W2V, LEADER, W2V_TEXT
from word_vectors import FileType
from word_vectors.write import write, write_glove, write_w2v, write_w2v_text, write_leader


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


def test_save_w2v_text(w, wv, file_name):
    write_w2v_text(file_name, w, wv)
    gold = open(DATA / W2V_TEXT).read()
    mine = open(file_name).read()
    assert mine == gold


def test_save_leader(w, wv, file_name):
    write_leader(file_name, w, wv)
    gold = open(DATA / LEADER, "rb").read()
    mine = open(file_name, "rb").read()
    assert mine == gold


def test_write():
    with patch("word_vectors.write_module.write_leader") as leader_patch, patch(
        "word_vectors.write_module.write_w2v"
    ) as w2v_patch, patch("word_vectors.write_module.write_w2v_text") as w2v_text_patch, patch(
        "word_vectors.write_module.write_glove"
    ) as glove_patch:
        gold_mapping = {
            FileType.GLOVE: glove_patch,
            FileType.W2V: w2v_patch,
            FileType.W2V_TEXT: w2v_text_patch,
            FileType.LEADER: leader_patch,
        }
        file_type = random.choice([FileType.GLOVE, FileType.W2V, FileType.W2V_TEXT, FileType.LEADER])
        write(None, None, None, file_type, None)
        for t, p in gold_mapping.items():
            if file_type is t:
                p.assert_called_once()
            else:
                p.assert_not_called()
