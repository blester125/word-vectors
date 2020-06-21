import struct
import random
from copy import deepcopy
from itertools import chain
from operator import itemgetter
from pathlib import Path
import pytest
import numpy as np
from word_vectors import FileType, DENSE_MAGIC_NUMBER
from word_vectors.read import (
    read,
    read_with_vocab,
    read_glove,
    read_w2v,
    read_w2v_text,
    read_dense,
    sniff,
    GLOVE_BIN,
    read_dense_header,
    verify_dense,
)
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
    rand_str,
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


@pytest.fixture
def correct_header():
    vsz, dsz, mxlen = np.random.randint(0, 20, size=(3,))
    header = struct.pack("<QQQQ", DENSE_MAGIC_NUMBER, vsz, dsz, mxlen)
    return header, vsz, dsz, mxlen


@pytest.fixture
def wrong_header():
    vsz, dsz, mxlen = np.random.randint(0, 20, size=(3,))
    magic = DENSE_MAGIC_NUMBER
    while magic == DENSE_MAGIC_NUMBER:
        magic = np.random.randint(10, 100000)
    header = struct.pack("<QQQQ", magic, vsz, dsz, mxlen)
    return header, vsz, dsz, mxlen


def sample(vocab, vectors, p):
    def inner_sample(vocab, vectors, p):
        new_vocab = {}
        new_vectors = []
        for word, idx in vocab.items():
            if np.random.rand() < p:
                new_vocab[word] = len(new_vocab)
                new_vectors.append(vectors[idx])
        return new_vocab, new_vectors

    new_vectors = []
    while not new_vectors:
        new_vocab, new_vectors = inner_sample(vocab, vectors, p)

    return new_vocab, np.vstack(new_vectors)


def split(vocab, vectors, p):
    def inner_split(vocab, vectors, p):
        new_vocab = {}
        new_vectors = []
        extra_vocab = {}
        extra_vectors = []
        for word, idx in vocab.items():
            if np.random.rand() < p:
                new_vocab[word] = len(new_vocab)
                new_vectors.append(vectors[idx])
            else:
                extra_vocab[word] = len(extra_vocab)
                extra_vectors.append(vectors[idx])
        return new_vocab, new_vectors, extra_vocab, extra_vectors

    new_vectors = []
    extra_vectors = []
    while not (new_vectors and extra_vectors):
        new_vocab, new_vectors, extra_vocab, extra_vectors = inner_split(vocab, vectors, p)

    return new_vocab, np.vstack(new_vectors), extra_vocab, np.vstack(extra_vectors)


def test_read_dense_header(correct_header):
    header, gvsz, gdsz, gmxlen = correct_header
    vsz, dsz, mxlen = read_dense_header(header)
    assert vsz == gvsz
    assert dsz == gdsz
    assert mxlen == gmxlen


def test_read_dense_header_errors(wrong_header):
    header, *_ = wrong_header
    with pytest.raises(ValueError):
        read_dense_header(header)


def test_verify_dense_correct(correct_header):
    header, *_ = correct_header
    assert verify_dense(header)


def test_verify_dense_wrong(wrong_header):
    header, *_ = wrong_header
    assert not verify_dense(header)


def test_read_dense_header_longer(correct_header):
    header, gvsz, gdsz, gmxlen = correct_header
    header = header + rand_str().encode("utf-8")
    vsz, dsz, mxlen = read_dense_header(header)
    assert vsz == gvsz
    assert dsz == gdsz
    assert mxlen == gmxlen


def test_read_dense_header_errors_longer(wrong_header):
    header, *_ = wrong_header
    header = header + rand_str().encode("utf-8")
    with pytest.raises(ValueError):
        read_dense_header(header)


def test_verify_dense_correct_longer(correct_header):
    header, *_ = correct_header
    header = header + rand_str().encode("utf-8")
    assert verify_dense(header)


def test_verify_dense_wrong_longer(wrong_header):
    header, *_ = wrong_header
    header = header + rand_str().encode("utf-8")
    assert not verify_dense(header)


def test_read(gold_vocab, gold_vectors):
    data = random.choice([GLOVE, W2V, DENSE, W2V_TEXT])
    w, wv = read(str(DATA / data))
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_with_vocab(gold_vocab, gold_vectors):
    gold_vocab, gold_vectors = sample(gold_vocab, gold_vectors, 0.5)
    data = random.choice([GLOVE, W2V, DENSE, W2V_TEXT])
    v, wv = read_with_vocab(str(DATA / data), gold_vocab)
    assert v == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_with_vocab_extra(gold_vocab, gold_vectors):
    user_vocab, user_vectors, extra_vocab, extra_vectors = split(gold_vocab, gold_vectors, 0.5)
    gold_vocab = {}
    gold_vectors = np.concatenate([user_vectors, extra_vectors], axis=0)
    for idx, (word, _) in enumerate(
        chain(sorted(user_vocab.items(), key=itemgetter(1)), sorted(extra_vocab.items(), key=itemgetter(1)))
    ):
        gold_vocab[word] = idx

    data = random.choice([GLOVE, W2V, DENSE, W2V_TEXT])
    v, wv = read_with_vocab(str(DATA / data), user_vocab, keep_extra=True)
    assert v == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_pathlib(gold_vocab, gold_vectors):
    data = random.choice([GLOVE, W2V, DENSE, W2V_TEXT])
    v, wv = read(DATA / data)
    assert v == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_with_vocab_pathlib(gold_vocab, gold_vectors):
    gold_vocab, gold_vectors = sample(gold_vocab, gold_vectors, 0.5)
    data = random.choice([GLOVE, W2V, DENSE, W2V_TEXT])
    v, wv = read_with_vocab(DATA / data, gold_vocab)
    assert v == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_with_vocab_extra_pathlib(gold_vocab, gold_vectors):
    user_vocab, user_vectors, extra_vocab, extra_vectors = split(gold_vocab, gold_vectors, 0.5)
    gold_vocab = {}
    gold_vectors = np.concatenate([user_vectors, extra_vectors], axis=0)
    for idx, (word, _) in enumerate(
        chain(sorted(user_vocab.items(), key=itemgetter(1)), sorted(extra_vocab.items(), key=itemgetter(1)))
    ):
        gold_vocab[word] = idx

    data = random.choice([GLOVE, W2V, DENSE, W2V_TEXT])
    v, wv = read_with_vocab(DATA / data, user_vocab, keep_extra=True)
    assert v == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_opened(gold_vocab, gold_vectors):
    data = random.choice([GLOVE, W2V, DENSE, W2V_TEXT])
    mode = "r" if data == GLOVE or data == W2V_TEXT else "rb"
    w, wv = read(open(DATA / data, mode))
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_with_vocab_opened(gold_vocab, gold_vectors):
    gold_vocab, gold_vectors = sample(gold_vocab, gold_vectors, 0.5)
    data = random.choice([GLOVE, W2V, DENSE, W2V_TEXT])
    mode = "r" if data in (GLOVE, W2V_TEXT) else "rb"
    v, wv = read_with_vocab(open(DATA / data, mode), gold_vocab)
    assert v == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_with_vocab_extra_opened(gold_vocab, gold_vectors):
    user_vocab, user_vectors, extra_vocab, extra_vectors = split(gold_vocab, gold_vectors, 0.5)
    gold_vocab = {}
    gold_vectors = np.concatenate([user_vectors, extra_vectors], axis=0)
    for idx, (word, _) in enumerate(
        chain(sorted(user_vocab.items(), key=itemgetter(1)), sorted(extra_vocab.items(), key=itemgetter(1)))
    ):
        gold_vocab[word] = idx

    data = random.choice([GLOVE, W2V, DENSE, W2V_TEXT])
    mode = "r" if data in (GLOVE, W2V_TEXT) else "rb"
    v, wv = read_with_vocab(open(DATA / data, mode), user_vocab, keep_extra=True)
    assert v == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_dupped(dupped_vocab, dupped_vectors):
    data = random.choice([GLOVE_DUPPED, W2V_DUPPED, DENSE_DUPPED])
    w, wv = read(str(DATA / data))
    assert w == dupped_vocab
    np.testing.assert_allclose(wv, dupped_vectors)


def test_read_dupped_with_vocab(dupped_vocab, dupped_vectors):
    data = random.choice([GLOVE_DUPPED, W2V_DUPPED, DENSE_DUPPED])
    gold_vocab, gold_vectors = sample(dupped_vocab, dupped_vectors, 0.5)
    w, wv = read_with_vocab(str(DATA / data), gold_vocab)
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_dupped_with_vocab_with_extra(dupped_vocab, dupped_vectors):
    data = random.choice([GLOVE_DUPPED, W2V_DUPPED, DENSE_DUPPED])
    user_vocab, user_vectors, extra_vocab, extra_vectors = split(dupped_vocab, dupped_vectors, 0.5)
    gold_vocab = {}
    gold_vectors = np.concatenate([user_vectors, extra_vectors], axis=0)
    for idx, (word, _) in enumerate(
        chain(sorted(user_vocab.items(), key=itemgetter(1)), sorted(extra_vocab.items(), key=itemgetter(1)))
    ):
        gold_vocab[word] = idx

    w, wv = read_with_vocab(str(DATA / data), user_vocab, keep_extra=True)
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_dupped_pathlib(dupped_vocab, dupped_vectors):
    data = random.choice([GLOVE_DUPPED, W2V_DUPPED, DENSE_DUPPED])
    w, wv = read(DATA / data)
    assert w == dupped_vocab
    np.testing.assert_allclose(wv, dupped_vectors)


def test_read_dupped_with_vocab_pathlib(dupped_vocab, dupped_vectors):
    data = random.choice([GLOVE_DUPPED, W2V_DUPPED, DENSE_DUPPED])
    gold_vocab, gold_vectors = sample(dupped_vocab, dupped_vectors, 0.5)
    w, wv = read_with_vocab(DATA / data, gold_vocab)
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_dupped_with_vocab_with_extra_pathlib(dupped_vocab, dupped_vectors):
    data = random.choice([GLOVE_DUPPED, W2V_DUPPED, DENSE_DUPPED])
    user_vocab, user_vectors, extra_vocab, extra_vectors = split(dupped_vocab, dupped_vectors, 0.5)
    gold_vocab = {}
    gold_vectors = np.concatenate([user_vectors, extra_vectors], axis=0)
    for idx, (word, _) in enumerate(
        chain(sorted(user_vocab.items(), key=itemgetter(1)), sorted(extra_vocab.items(), key=itemgetter(1)))
    ):
        gold_vocab[word] = idx

    w, wv = read_with_vocab(DATA / data, user_vocab, keep_extra=True)
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_dupped_opened(dupped_vocab, dupped_vectors):
    data = random.choice([GLOVE_DUPPED, W2V_DUPPED, DENSE_DUPPED])
    mode = "r" if data == GLOVE_DUPPED else "rb"
    w, wv = read(open(DATA / data, mode))
    assert w == dupped_vocab
    np.testing.assert_allclose(wv, dupped_vectors)


def test_read_dupped_with_vocab_opened(dupped_vocab, dupped_vectors):
    data = random.choice([GLOVE_DUPPED, W2V_DUPPED, DENSE_DUPPED])
    mode = "r" if data == GLOVE_DUPPED else "rb"
    gold_vocab, gold_vectors = sample(dupped_vocab, dupped_vectors, 0.5)
    w, wv = read_with_vocab(open(DATA / data, mode), gold_vocab)
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


def test_read_dupped_with_vocab_with_extra_opened(dupped_vocab, dupped_vectors):
    data = random.choice([GLOVE_DUPPED, W2V_DUPPED, DENSE_DUPPED])
    mode = "r" if data == GLOVE_DUPPED else "rb"
    user_vocab, user_vectors, extra_vocab, extra_vectors = split(dupped_vocab, dupped_vectors, 0.5)
    gold_vocab = {}
    gold_vectors = np.concatenate([user_vectors, extra_vectors], axis=0)
    for idx, (word, _) in enumerate(
        chain(sorted(user_vocab.items(), key=itemgetter(1)), sorted(extra_vocab.items(), key=itemgetter(1)))
    ):
        gold_vocab[word] = idx

    w, wv = read_with_vocab(open(DATA / data, mode), user_vocab, keep_extra=True)
    assert w == gold_vocab
    np.testing.assert_allclose(wv, gold_vectors)


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
    with open(DATA / W2V, "rb") as f:
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
    with open(DATA / W2V_DUPPED, "rb") as f:
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
    with open(DATA / DENSE, "rb") as f:
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
    with open(DATA / DENSE_DUPPED, "rb") as f:
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
