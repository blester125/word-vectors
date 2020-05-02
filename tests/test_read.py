import random
from pathlib import Path
import pytest
import numpy as np
from word_vectors.read import read, read_glove, read_w2v, read_w2v_text, read_dense, sniff, FileType, GLOVE_BIN
from utils import (
    vocab,
    vectors,
    DATA,
    GLOVE,
    W2V,
    W2V_TEXT,
    DENSE,
    dupped_vectors as dupped_vects,
    dupped_vocab as dupped_vs,
    GLOVE_DUPPED,
    W2V_DUPPED,
    W2V_TEXT_DUPPED,
    DENSE_DUPPED,
)


@pytest.fixture
def gold_vocab():
    return vocab


@pytest.fixture
def gold_vectors():
    return vectors


@pytest.fixture
def dupped_vocab():
    return dupped_vs


@pytest.fixture
def dupped_vectors():
    return dupped_vects


def test_read(gold_vocab, gold_vectors):
    data = random.choice([GLOVE, W2V, DENSE, W2V_TEXT])
    w, wv = read(str(DATA / data))
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_pathlib(gold_vocab, gold_vectors):
    data = random.choice([GLOVE, W2V, DENSE, W2V_TEXT])
    w, wv = read(DATA / data)
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_opened(gold_vocab, gold_vectors):
    data = random.choice([GLOVE, W2V, DENSE, W2V_TEXT])
    mode = "r" if data == GLOVE or data == W2V_TEXT else "rb"
    w, wv = read(open(DATA / data, mode))
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_dupped(dupped_vocab, dupped_vectors):
    data = random.choice([GLOVE_DUPPED, W2V_DUPPED, DENSE_DUPPED])
    w, wv = read(str(DATA / data))
    assert w == dupped_vocab
    np.testing.assert_allclose(wv, dupped_vectors)


def test_read_dupped_pathlib(dupped_vocab, dupped_vectors):
    data = random.choice([GLOVE_DUPPED, W2V_DUPPED, DENSE_DUPPED])
    w, wv = read(DATA / data)
    assert w == dupped_vocab
    np.testing.assert_allclose(wv, dupped_vectors)


def test_read_dupped_opened(dupped_vocab, dupped_vectors):
    data = random.choice([GLOVE_DUPPED, W2V_DUPPED, DENSE_DUPPED])
    mode = "r" if data == GLOVE_DUPPED else "rb"
    w, wv = read(open(DATA / data, mode))
    assert w == dupped_vocab
    np.testing.assert_allclose(wv, dupped_vectors)


def test_read_glove(gold_vocab, gold_vectors):
    w, wv = read_glove(str(DATA / GLOVE))
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_glove_pathlib(gold_vocab, gold_vectors):
    w, wv = read_glove(DATA / GLOVE)
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_glove_opened(gold_vocab, gold_vectors):
    with open(DATA / GLOVE) as f:
        w, wv = read_glove(f)
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_glove_dupped(dupped_vocab, dupped_vectors):
    w, wv = read_glove(str(DATA / GLOVE_DUPPED))
    assert w == dupped_vocab
    np.testing.assert_allclose(wv, dupped_vectors)


def test_read_glove_dupped_pathlib(dupped_vocab, dupped_vectors):
    w, wv = read_glove(DATA / GLOVE_DUPPED)
    assert w == dupped_vocab
    np.testing.assert_allclose(wv, dupped_vectors)


def test_read_glove_dupped_opened(dupped_vocab, dupped_vectors):
    with open(DATA / GLOVE_DUPPED) as f:
        w, wv = read_glove(f)
    assert w == dupped_vocab
    np.testing.assert_allclose(wv, dupped_vectors)


def test_read_w2v(gold_vocab, gold_vectors):
    w, wv = read_w2v(str(DATA / W2V))
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_w2v_pathlib(gold_vocab, gold_vectors):
    w, wv = read_w2v(DATA / W2V)
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_w2v_opened(gold_vocab, gold_vectors):
    with open(DATA / W2V, 'rb') as f:
        w, wv = read_w2v(f)
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_w2v_dupped(dupped_vocab, dupped_vectors):
    w, wv = read_w2v(str(DATA / W2V_DUPPED))
    assert w == dupped_vocab
    np.testing.assert_allclose(wv, dupped_vectors)


def test_read_w2v_dupped_pathlib(dupped_vocab, dupped_vectors):
    w, wv = read_w2v(DATA / W2V_DUPPED)
    assert w == dupped_vocab
    np.testing.assert_allclose(wv, dupped_vectors)


def test_read_w2v_dupped_opened(dupped_vocab, dupped_vectors):
    with open(DATA / W2V_DUPPED, 'rb') as f:
        w, wv = read_w2v(f)
    assert w == dupped_vocab
    np.testing.assert_allclose(wv, dupped_vectors)


def test_read_w2v_text(gold_vocab, gold_vectors):
    w, wv = read_w2v_text(str(DATA / W2V_TEXT))
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_w2v_text_pathlib(gold_vocab, gold_vectors):
    w, wv = read_w2v_text(DATA / W2V_TEXT)
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_w2v_text_opened(gold_vocab, gold_vectors):
    with open(DATA / W2V_TEXT) as f:
        w, wv = read_w2v_text(f)
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_w2v_text_dupped(dupped_vocab, dupped_vectors):
    w, wv = read_w2v_text(str(DATA / W2V_TEXT_DUPPED))
    assert w == dupped_vocab
    np.testing.assert_allclose(wv, dupped_vectors)


def test_read_w2v_text_dupped_pathlib(dupped_vocab, dupped_vectors):
    w, wv = read_w2v_text(DATA / W2V_TEXT_DUPPED)
    assert w == dupped_vocab
    np.testing.assert_allclose(wv, dupped_vectors)


def test_read_w2v_text_dupped_opened(dupped_vocab, dupped_vectors):
    with open(DATA / W2V_TEXT_DUPPED) as f:
        w, wv = read_w2v_text(f)
    assert w == dupped_vocab
    np.testing.assert_allclose(wv, dupped_vectors)


def test_read_dense(gold_vocab, gold_vectors):
    w, wv = read_dense(str(DATA / DENSE))
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_dense_pathlib(gold_vocab, gold_vectors):
    w, wv = read_dense(DATA / DENSE)
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_dense_opened(gold_vocab, gold_vectors):
    with open(DATA / DENSE, 'rb') as f:
        w, wv = read_dense(f)
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_dense_dupped(dupped_vocab, dupped_vectors):
    w, wv = read_dense(str(DATA / DENSE_DUPPED))
    assert w == dupped_vocab
    np.testing.assert_allclose(wv, dupped_vectors)


def test_read_dense_dupped_pathlib(dupped_vocab, dupped_vectors):
    w, wv = read_dense(DATA / DENSE_DUPPED)
    assert w == dupped_vocab
    np.testing.assert_allclose(wv, dupped_vectors)


def test_read_dense_dupped_opened(dupped_vocab, dupped_vectors):
    with open(DATA / DENSE_DUPPED, 'rb') as f:
        w, wv = read_dense(f)
    assert w == dupped_vocab
    np.testing.assert_allclose(wv, dupped_vectors)


def test_glove_regex_weird_start():
    ex = "<user< -0.4532 23.123\n".encode("utf-8")
    match = GLOVE_BIN.search(ex)
    assert match is not None


def test_sniff_glove():
    x = sniff(DATA / GLOVE)
    assert x is FileType.GLOVE


def test_sniff_w2v():
    x = sniff(DATA / W2V)
    assert x is FileType.W2V


def test_sniff_w2v_text():
    x = sniff(DATA / W2V_TEXT)
    assert x is FileType.W2V_TEXT


def test_sniff_dense():
    x = sniff(DATA / DENSE)
    assert x is FileType.DENSE
