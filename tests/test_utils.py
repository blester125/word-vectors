import io
import random
import string
import pathlib
from io import StringIO
from word_vectors import FileType
from word_vectors.utils import find_max, is_binary, find_space, bookmark, padded_bytes, to_vocab, create_output_path
from utils import DATA, GLOVE, W2V, W2V_TEXT, DENSE, rand_str


def test_enum_parse():
    gold_mapping = {
        "glove": FileType.GLOVE,
        "GloVe": FileType.GLOVE,
        "w2v_text": FileType.W2V_TEXT,
        "w2v-text": FileType.W2V_TEXT,
        "w2v": FileType.W2V,
        "dense": FileType.DENSE,
        "fasttext": FileType.FASTTEXT,
        "fast-text": FileType.FASTTEXT,
        "fast_text": FileType.FASTTEXT,
        "numberbatch": FileType.NUMBERBATCH,
    }
    for s, t in gold_mapping.items():
        assert FileType.from_string(s) is t


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
    file_to_gold = {DATA / GLOVE: False, DATA / W2V: True, DATA / W2V_TEXT: False, DATA / DENSE: True}
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


def test_to_vocab():
    vocab = list("ABCDEFGHIJKLMNOP")
    random.shuffle(vocab)
    d = {k: i for i, k in enumerate(vocab)}
    assert to_vocab(vocab) == d


def test_padded_bytes():
    len_ = random.randint(5, 11)
    text = "".join(random.choice(string.ascii_lowercase) for _ in range(len_))
    len_ = random.randint(len(text), len(text) + 10)
    res = padded_bytes(text, len_)
    assert len(res) == len_
    assert res[: len(text)] == text.encode("utf-8")


def test_padded_bytes_longer():
    len_ = random.randint(5, 11)
    text = "".join(random.choice(string.ascii_lowercase) for _ in range(len_))
    len_ -= 1
    res = padded_bytes(text, len_)
    assert len(res) == len(text)
    assert len(res) > len_
    assert res == text.encode("utf-8")


def test_create_output_path():
    ext = rand_str()
    base = rand_str()
    file_type = random.choice(list(FileType))
    path = f"{ext}.{base}"
    gold = f"{ext}.{file_type}"
    assert create_output_path(path, file_type) == gold


def test_create_output_path_pathlib():
    ext = rand_str()
    base = rand_str()
    file_type = random.choice(list(FileType))
    path = pathlib.Path(f"{ext}.{base}")
    gold = f"{ext}.{file_type}"
    assert create_output_path(path, file_type) == gold


def test_create_output_path_open():
    ext = rand_str()
    base = rand_str()
    file_type = random.choice(list(FileType))
    path = f"{ext}.{base}"
    path_file = io.StringIO(path)
    path_file.name = path
    gold = f"{ext}.{file_type}"
    assert create_output_path(path_file, file_type) == gold


def test_create_output_path_open_bytes():
    ext = rand_str()
    base = rand_str()
    file_type = random.choice(list(FileType))
    path = f"{ext}.{base}"
    path_file = io.BytesIO(path.encode("utf-8"))
    path_file.name = path
    gold = f"{ext}.{file_type}"
    assert create_output_path(path_file, file_type) == gold
