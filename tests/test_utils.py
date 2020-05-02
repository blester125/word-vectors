from io import StringIO
from word_vectors.utils import find_max, is_binary, find_space, bookmark
from utils import (
    DATA, GLOVE, W2V, W2V_TEXT, DENSE, rand_str
)


def test_max():
    words = ["A" * i for i in range(10)]
    gold_len = 9
    max_len = find_max(words)
    assert max_len == gold_len


def test_max_bytes():
    words = ["a", "🙄", "c"]
    str_max = len(max(words, key=lambda x: len(x)))
    gold = 4
    res = find_max(words)
    assert res == gold
    assert res > str_max


def test_is_binary():
    file_to_gold = {
        DATA / GLOVE: False,
        DATA / W2V: True,
        DATA / W2V_TEXT: False,
        DATA / DENSE: True
    }
    for file_name, gold in file_to_gold.items():
        assert is_binary(file_name) == gold


def test_find_space():
    gold = rand_str()
    gold_offset = len(gold.encode("utf-8")) + 1
    extra = rand_str()
    text = f"{gold} {extra}".encode("utf-8")
    word, offset = find_space(text, offset=0)
    assert word == gold
    assert offset == gold_offset


def test_find_space_with_offset():
    before = rand_str()
    start = len(before.encode("utf-8")) + 1
    gold = rand_str()
    gold_offset = start + len(gold.encode("utf-8")) + 1
    after = rand_str()
    text = f"{before} {gold} {after}".encode("utf-8")
    word, offset = find_space(text, offset=start)


def test_bookmark():
    data = StringIO("bad\ngood\nbad")
    _ = data.readline()
    with bookmark(data):
        line = data.readline()
    next_line = data.readline()
    assert next_line == line
